// Copyright 2023 The Manifold Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

#include "meshbool/meshbool.h"
#include "meshbool/optional_assert.h"

namespace manifold {

/**
 * View for Vec, can perform offset operation.
 * This will be invalidated when the original vector is dropped or changes
 * length. Roughly equivalent to std::span<T> from c++20
 */
template <typename T>
class VecView {
 public:
  using Iter = T*;
  using IterC = const T*;

  VecView() : ptr_(nullptr), size_(0) {}

  VecView(T* ptr, size_t size) : ptr_(ptr), size_(size) {}

  VecView(const std::vector<std::remove_cv_t<T>>& v)
      : ptr_(v.data()), size_(v.size()) {}

  VecView(const VecView& other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
  }

  VecView& operator=(const VecView& other) {
    ptr_ = other.ptr_;
    size_ = other.size_;
    return *this;
  }

  // allows conversion to a const VecView
  operator VecView<const T>() const { return {ptr_, size_}; }

  inline const T& operator[](size_t i) const {
    ASSERT(i < size_, std::out_of_range("Vec out of range"));
    return ptr_[i];
  }

  inline T& operator[](size_t i) {
    ASSERT(i < size_, std::out_of_range("Vec out of range"));
    return ptr_[i];
  }

  IterC cbegin() const { return ptr_; }
  IterC cend() const { return ptr_ + size_; }

  IterC begin() const { return cbegin(); }
  IterC end() const { return cend(); }

  Iter begin() { return ptr_; }
  Iter end() { return ptr_ + size_; }

  const T& front() const {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the front of an empty vector"));
    return ptr_[0];
  }

  const T& back() const {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the back of an empty vector"));
    return ptr_[size_ - 1];
  }

  T& front() {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the front of an empty vector"));
    return ptr_[0];
  }

  T& back() {
    ASSERT(size_ != 0,
           std::out_of_range("Attempt to take the back of an empty vector"));
    return ptr_[size_ - 1];
  }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  VecView<T> view(size_t offset = 0,
                  size_t length = std::numeric_limits<size_t>::max()) {
    if (length == std::numeric_limits<size_t>::max())
      length = this->size_ - offset;
    ASSERT(offset + length <= this->size_,
           std::out_of_range("Vec::view out of range"));
    return VecView<T>(this->ptr_ + offset, length);
  }

  VecView<const T> cview(
      size_t offset = 0,
      size_t length = std::numeric_limits<size_t>::max()) const {
    if (length == std::numeric_limits<size_t>::max())
      length = this->size_ - offset;
    ASSERT(offset + length <= this->size_,
           std::out_of_range("Vec::cview out of range"));
    return VecView<const T>(this->ptr_ + offset, length);
  }

  VecView<const T> view(
      size_t offset = 0,
      size_t length = std::numeric_limits<size_t>::max()) const {
    return cview(offset, length);
  }

  T* data() { return this->ptr_; }

  const T* data() const { return this->ptr_; }

#ifdef MANIFOLD_DEBUG
  void Dump() const {
    std::cout << "Vec = " << std::endl;
    for (size_t i = 0; i < size(); ++i) {
      std::cout << i << ", " << ptr_[i] << ", " << std::endl;
    }
    std::cout << std::endl;
  }
#endif

 protected:
  T* ptr_ = nullptr;
  size_t size_ = 0;
};

template <typename T>
class VecWrapper {
 private:
  ::rust::RefMut<rust::std::vec::Vec<T>> internal;

 public:
  inline VecWrapper(::rust::RefMut<rust::std::vec::Vec<T>> i) : internal(i) {}
  // inline VecWrapper() : internal(rust::std::vec::Vec<T>::new_()) {}
  VecWrapper() = delete;
  inline size_t size() const { return this->internal.len(); }

  template <typename Tj>
  class Iterator {
   private:
    Tj* m_ptr;

   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = Tj;
    using difference_type = std::ptrdiff_t;
    using pointer = Tj*;
    using reference = Tj&;

    Iterator() = delete;
    Iterator(pointer p) : m_ptr(p) {}

    // Dereference
    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }

    Iterator& operator+(size_t i) {
      this->m_ptr + i;
      return *this;
    }

    // Prefix increment
    Iterator& operator++() {
      ++m_ptr;
      return *this;
    }

    // Postfix increment
    Iterator operator++(int) {
      Iterator tmp = *this;
      ++(*this);
      return tmp;
    }

    // Comparison
    friend bool operator==(const Iterator& a, const Iterator& b) {
      return a.m_ptr == b.m_ptr;
    }
    friend bool operator!=(const Iterator& a, const Iterator& b) {
      return a.m_ptr != b.m_ptr;
    }
  };

  // inline void insert(const Iter &iter, const T &value) {}
  inline void insert(const Iterator<T>& iter, std::initializer_list<T> list) {}

  Iterator<T> begin() { return this->data(); }
  Iterator<const T> begin() const { return this->data(); }
  Iterator<T> end() { return this->data() + this->size(); }
  Iterator<const T> end() const { return this->data() + this->size(); }

  inline VecWrapper& operator=(std::initializer_list<T> list) {
    this->clear();
    for (const T& v : list) {
      this->push_back(v);
    }
    return *this;
  }

  inline VecWrapper& operator=(VecView<T> view) {
    this->clear();
    for (const T& v : view) {
      this->push_back(v);
    }
    return *this;
  }

  inline T& operator[](size_t idx) { return *this->internal.get(idx).unwrap(); }
  inline const T& operator[](size_t idx) const {
    return *this->internal.get(idx).unwrap();
  }
  // inline T& operator[](size_t idx) { return this->data()[idx]; }
  // inline const T& operator[](size_t idx) const { return this->data()[idx]; }

  inline bool empty() const { return this->internal.is_empty(); }

  inline void push_back(const T& value) { this->internal.push(value); }
  inline void push_back(T&& value) { this->internal.push(value); }
  inline T* data() { return this->internal.as_mut_ptr(); }
  inline const T* data() const { return this->internal.as_ptr(); }
  inline T& front() { return this->data()[0]; }
  inline const T& front() const { return this->data()[0]; }
  inline T& back() { return this->data()[this->internal.len() - 1]; }
  inline const T& back() const {
    return this->data()[this->internal.len() - 1];
  }

  inline void clear() { this->internal.clear(); }
  inline void resize(size_t new_len) { this->internal.resize(new_len, T()); }
  inline void resize(size_t new_len, const T& value) {
    this->internal.resize(new_len, value);
  }

  operator std::vector<T>() const {
    std::vector<T> result;
    result.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) result.push_back((*this)[i]);
    return result;
  }

  operator const VecView<const T>() const {
    return VecView(this->data(), this->size());
  }
};

}  // namespace manifold
