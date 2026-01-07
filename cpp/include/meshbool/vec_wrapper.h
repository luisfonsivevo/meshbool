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
