// libpli_ane: Neural Engine client for PLI (15/10/2026).
// Takes ane/Program.mm from Splash 1.3.2-m1 (paperniuk/splash, branch apple7-m1-1.3.2, commit 5c484b2,
// Apache-2.0, see LICENSE-splash) without modifying it; this file adds the Metal shim (metal/MetalBackend.hpp)
// and a C API called through ctypes from PLI's Python.
//
// Synchronous evaluation (path a): the CPU raises the event at `wait`, the service evaluates then raises it at `signal`,
// the CPU waits for that signal. The time returned goes from queueing to the signal.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include "ane/Program.hpp"

#include <chrono>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace splash;

namespace splash::metal {

static id<MTLSharedEvent> native(const SharedEvent &event) { return (__bridge id<MTLSharedEvent>)event.nativeHandle(); }

void SharedEvent::signal(uint64_t value) const noexcept { native(*this).signaledValue = value; }
uint64_t SharedEvent::value() const noexcept { return native(*this).signaledValue; }

void SharedEvent::notify(uint64_t value, std::function<void()> callback) const {
  static MTLSharedEventListener *listener =
      [[MTLSharedEventListener alloc] initWithDispatchQueue:dispatch_queue_create("pli.ane.events", DISPATCH_QUEUE_SERIAL)];
  auto held = std::make_shared<std::function<void()>>(std::move(callback));
  [native(*this) notifyListener:listener
                        atValue:value
                          block:^(id<MTLSharedEvent>, uint64_t) {
                            (*held)();
                          }];
}

SharedEvent newSharedEvent() {
  static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (!device) throw std::runtime_error("no Metal device");
  id<MTLSharedEvent> event = [device newSharedEvent];
  if (!event) throw std::runtime_error("no shared event");
  return SharedEvent(std::shared_ptr<void>((void *)CFBridgingRetain(event), [](void *p) { CFRelease(p); }));
}

} // namespace splash::metal

namespace {

thread_local std::string lastError;
metal::MetalBackend backend;

struct Prog {
  std::unique_ptr<ane::Program> program;
  metal::SharedEvent event;
  uint64_t value = 0;
};

struct Bind {
  ane::Program::Binding binding;
};

struct Done {
  std::mutex mutex;
  std::condition_variable cv;
  int result = -1; // -1: not yet, 0: failure, 1: success
};

ane::Program::Limits limits(double seconds) {
  return {std::chrono::duration_cast<AwakeClock::duration>(std::chrono::duration<double>(seconds)),
          [] { return false; }};
}

int copyOut(const std::string &text, char *out, size_t size) {
  if (out && size) {
    const size_t n = std::min(size - 1, text.size());
    std::memcpy(out, text.data(), n);
    out[n] = 0;
  }
  return static_cast<int>(text.size());
}

template <class F> auto guarded(F &&f, decltype(f()) failed) -> decltype(f()) {
  try {
    return f();
  } catch (const std::exception &error) {
    lastError = error.what();
    return failed;
  } catch (...) {
    lastError = "unknown error";
    return failed;
  }
}

} // namespace

extern "C" {

const char *pli_ane_error() { return lastError.c_str(); }

// PLI (16/10/2026): Neural Engine block (1 to 3; 0: the service chooses) of the programs built afterwards.
void pli_ane_next_block(int block) { ane::setNextBlock(block); }

// Empty string when the Neural Engine is reachable; otherwise the reason.
int pli_ane_unavailable(char *out, size_t size) {
  const std::optional<std::string> why = ane::unavailable();
  return copyOut(why ? *why : std::string(), out, size);
}

void *pli_ane_surface(uint32_t rows, uint32_t width, int int8) {
  return guarded(
      [&]() -> void * {
        return new ane::Surface(
            ane::Surface::create(backend, rows, width, int8 ? ane::Surface::Element::Int8 : ane::Surface::Element::Float16));
      },
      nullptr);
}
void pli_ane_surface_free(void *surface) { delete static_cast<ane::Surface *>(surface); }
void *pli_ane_surface_base(void *surface) {
  return IOSurfaceGetBaseAddress(static_cast<IOSurfaceRef>(static_cast<ane::Surface *>(surface)->surface.get()));
}
uint32_t pli_ane_surface_stride(void *surface) { return static_cast<ane::Surface *>(surface)->strideBytes; }
uint64_t pli_ane_surface_size(void *surface) {
  return IOSurfaceGetAllocSize(static_cast<IOSurfaceRef>(static_cast<ane::Surface *>(surface)->surface.get()));
}
int pli_ane_surface_type(void *surface, uint32_t rows, uint32_t width, char *out, size_t size) {
  return copyOut(static_cast<ane::Surface *>(surface)->bufferType(rows, width), out, size);
}
int pli_ane_surface_strides(void *surface, uint32_t rows, char *out, size_t size) {
  return copyOut(static_cast<ane::Surface *>(surface)->strides(rows), out, size);
}

// Weight blob of an fp16 constant (BLOBFILE ... offset = 64).
size_t pli_ane_blob(const uint16_t *values, size_t count, uint8_t *out, size_t size) {
  const std::vector<uint8_t> blob =
      ane::constantBlob(std::span<const _Float16>(reinterpret_cast<const _Float16 *>(values), count));
  if (out && size >= blob.size()) std::memcpy(out, blob.data(), blob.size());
  return blob.size();
}

void *pli_ane_program(const char *mil, const uint8_t *weights, size_t size, const char *cache, double limit) {
  return guarded(
      [&]() -> void * {
        auto prog = std::make_unique<Prog>();
        prog->program = std::make_unique<ane::Program>(std::string_view(mil), std::span<const uint8_t>(weights, size),
                                                       limits(limit), std::filesystem::path(cache));
        prog->event = metal::newSharedEvent();
        return prog.release();
      },
      nullptr);
}
void pli_ane_program_free(void *prog) { delete static_cast<Prog *>(prog); }

int pli_ane_procedure(void *prog, const char *function) {
  return guarded([&]() -> int { return static_cast<int>(static_cast<Prog *>(prog)->program->procedure(function)); }, -1);
}

int pli_ane_inputs(void *prog, int procedure, char *out, size_t size) {
  return guarded(
      [&]() -> int {
        std::string text;
        for (const std::string &name : static_cast<Prog *>(prog)->program->inputs(static_cast<uint32_t>(procedure)))
          text += (text.empty() ? "" : "\n") + name;
        return copyOut(text, out, size);
      },
      -1);
}

void *pli_ane_bind(void *prog, int procedure, void **inputs, int count, void *output) {
  return guarded(
      [&]() -> void * {
        std::vector<ane::Surface> surfaces;
        for (int i = 0; i < count; ++i) surfaces.push_back(*static_cast<ane::Surface *>(inputs[i]));
        return new Bind{static_cast<Prog *>(prog)->program->bind(static_cast<uint32_t>(procedure), surfaces,
                                                                *static_cast<ane::Surface *>(output))};
      },
      nullptr);
}
void pli_ane_bind_free(void *bind) { delete static_cast<Bind *>(bind); }

// One evaluation, waited for by the CPU. Returns the milliseconds from queueing to the signal, -1 on failure
// (reason in pli_ane_error), -2 if the timeout is exceeded.
double pli_ane_eval(void *progp, void *bindp, double timeout) {
  Prog *prog = static_cast<Prog *>(progp);
  Bind *bind = static_cast<Bind *>(bindp);
  const uint64_t wait = ++prog->value, signal = ++prog->value;
  auto done = std::make_shared<Done>();
  const auto start = std::chrono::steady_clock::now();
  prog->event.signal(wait);
  const bool queued = guarded(
      [&]() -> bool {
        prog->program->enqueue(bind->binding, prog->event, wait, signal, [done](bool ok) {
          std::lock_guard lock(done->mutex);
          done->result = ok ? 1 : 0;
          done->cv.notify_all();
        });
        return true;
      },
      false);
  if (!queued) return -1.0;
  const auto deadline = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                    std::chrono::duration<double>(timeout));
  while (prog->event.value() < signal) {
    {
      std::lock_guard lock(done->mutex);
      if (done->result == 0) {
        lastError = "ANE evaluation failed";
        prog->value = std::max(prog->value, signal);
        return -1.0;
      }
    }
    if (std::chrono::steady_clock::now() > deadline) {
      lastError = "ANE evaluation timed out";
      return -2.0;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

} // extern "C"
