#pragma once
// Shim of libpli_ane (PLI, 15/10/2026): replaces Splash 1.3.2-m1's metal/MetalBackend.hpp (Apache-2.0) for
// ane/Program.mm, which uses only MetalBackend::wrapSharedMemory, MetalBuffer and SharedEvent from it.
// The Metal buffer of a surface is not created here: on the PLI side, MLX puts it on the surface's address
// (allocator::make_buffer). The events are real MTLSharedEvent (the service shares them through their Mach port).

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>

namespace splash::metal {

class MetalBuffer final {
public:
  void *address = nullptr;
  uint64_t bytes = 0;
  std::shared_ptr<void> owner;
  [[nodiscard]] void *contents() const noexcept { return address; }
  [[nodiscard]] uint64_t sizeBytes() const noexcept { return bytes; }
  [[nodiscard]] uint64_t allocatedBytes() const noexcept { return bytes; }
};

class SharedEvent final {
public:
  SharedEvent() = default;
  explicit SharedEvent(std::shared_ptr<void> event) : event_(std::move(event)) {}
  [[nodiscard]] explicit operator bool() const noexcept { return bool(event_); }
  [[nodiscard]] void *nativeHandle() const noexcept { return event_.get(); }
  void signal(uint64_t value) const noexcept;
  [[nodiscard]] uint64_t value() const noexcept;
  void notify(uint64_t value, std::function<void()> callback) const;

private:
  std::shared_ptr<void> event_;
};

class MetalBackend final {
public:
  [[nodiscard]] MetalBuffer wrapSharedMemory(void *address, uint64_t bytes, std::shared_ptr<void> owner,
                                             std::string_view) {
    MetalBuffer buffer;
    buffer.address = address;
    buffer.bytes = bytes;
    buffer.owner = std::move(owner);
    return buffer;
  }
};

[[nodiscard]] SharedEvent newSharedEvent();

} // namespace splash::metal
