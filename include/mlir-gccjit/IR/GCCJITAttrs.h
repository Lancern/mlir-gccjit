// Copyright 2024 Schrodinger ZHU Yifan <i@zhuyi.fan>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MLIR_GCCJIT_IR_GCCJIT_OPS_ATTRS_H
#define MLIR_GCCJIT_IR_GCCJIT_OPS_ATTRS_H

#include "mlir-gccjit/IR/GCCJITOpsEnums.h"
#include "mlir-gccjit/IR/GCCJITTypes.h"

#define GET_ATTRDEF_CLASSES
#include "mlir-gccjit/IR/GCCJITOpsAttributes.h.inc"

namespace mlir::gccjit {
inline bool isUnitFnAttr(FnAttrEnum attr) {
  switch (attr) {
  case FnAttrEnum::AlwaysInline:
  case FnAttrEnum::Inline:
  case FnAttrEnum::NoInline:
  case FnAttrEnum::Used:
  case FnAttrEnum::Cold:
  case FnAttrEnum::ReturnsTwice:
  case FnAttrEnum::Pure:
  case FnAttrEnum::Const:
  case FnAttrEnum::Weak:
    return true;
  default:
    return false;
  }
}
inline bool isStringFnAttr(FnAttrEnum attr) {
  switch (attr) {
  case FnAttrEnum::Alias:
  case FnAttrEnum::Target:
    return true;
  default:
    return false;
  }
}

inline bool isIntArrayFnAttr(FnAttrEnum attr) {
  switch (attr) {
  case FnAttrEnum::Nonnull:
    return true;
  default:
    return false;
  }
}
} // namespace mlir::gccjit

#endif // MLIR_GCCJIT_IR_GCCJIT_OPS_ATTRS_H
