#pragma once
// Cale de libpli_ane (PLI, 15/10/2026) : remplace metal/MetalBackend.hpp de Splash 1.3.2-m1 (Apache-2.0) pour
// ane/Program.mm, qui n'en utilise que MetalBackend::wrapSharedMemory, MetalBuffer et SharedEvent.
// Le tampon Metal d'une surface n'est pas créé ici : côté PLI, MLX le pose sur l'adresse de la surface
// (allocator::make_buffer). Les événements sont de vrais MTLSharedEvent (le service les partage par leur port Mach).

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
