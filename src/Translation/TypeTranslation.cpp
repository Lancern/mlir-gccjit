#include "mlir-gccjit/Translation/TranslateToGCCJIT.h"

#include <climits>
#include <optional>

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/BuiltinTypes.h>

#include <libgccjit.h>

#include "mlir-gccjit/IR/GCCJITTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::gccjit {

size_t GCCJITTranslation::getTypeSize(Type type) {
  // gcc_jit_type_get_size only works for integer types now.
  if (mlir::isa<IntegerType, IntType>(type))
    return gcc_jit_type_get_size(convertType(type));

  if (mlir::isa<IndexType, PointerType>(type)) {
    // Let's pray size_t and pointers have the same size on users' machines.
    gcc_jit_type *sizeType =
        gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_SIZE_T);
    return gcc_jit_type_get_size(sizeType);
  }

  if (auto floatTy = mlir::dyn_cast<FloatType>(type)) {
    switch (floatTy.getKind()) {
    case GCC_JIT_TYPE_FLOAT:
      return 4;
    case GCC_JIT_TYPE_DOUBLE:
      return 8;
    // TODO: add getTypeSize support for long double type.
    default:
      llvm_unreachable("unsupported gccjit float type");
    }
  }
  if (mlir::isa<BFloat16Type>(type))
    return 2;
  if (mlir::isa<Float32Type>(type))
    return 4;
  if (mlir::isa<Float64Type>(type))
    return 8;

  // TODO: add getTypeSize support for struct type and union type.

  llvm_unreachable("unsupported gccjit object type");
}

void GCCJITTranslation::convertTypes(
    mlir::TypeRange types, llvm::SmallVector<gcc_jit_type *> &result) {
  for (auto type : types)
    result.push_back(convertType(type));
}

gcc_jit_type *GCCJITTranslation::convertType(mlir::Type type) {
  if (auto it = typeMap.find(type); it != typeMap.end())
    return it->second;
  auto *res =
      llvm::TypeSwitch<mlir::Type, gcc_jit_type *>(type)
          .Case([&](gccjit::LValueType t) {
            return convertType(t.getInnerType());
          })
          .Case([&](gccjit::PointerType t) {
            auto *pointee = convertType(t.getElementType());
            if (isa<gccjit::FuncType>(t.getElementType()))
              return pointee;
            if (auto qualified =
                    dyn_cast<gccjit::QualifiedType>(t.getElementType())) {
              if (qualified.getIsConst())
                if (auto intType =
                        dyn_cast<gccjit::IntType>(qualified.getElementType()))
                  if (intType.getKind() == GCC_JIT_TYPE_CHAR)
                    return gcc_jit_context_get_type(
                        ctxt, GCC_JIT_TYPE_CONST_CHAR_PTR);
            }
            return gcc_jit_type_get_pointer(pointee);
          })
          .Case([&](gccjit::FuncType t) {
            llvm::SmallVector<gcc_jit_type *> paramTypes;
            convertTypes(t.getInputs(), paramTypes);
            auto *returnType = convertType(t.getReturnType());
            return gcc_jit_context_new_function_ptr_type(
                ctxt, nullptr, returnType, paramTypes.size(), paramTypes.data(),
                t.isVarArg());
          })
          .Case([&](gccjit::QualifiedType t) {
            auto *res = convertType(t.getElementType());
            if (t.getIsConst())
              res = gcc_jit_type_get_const(res);
            if (t.getIsRestrict())
              res = gcc_jit_type_get_restrict(res);
            if (t.getIsVolatile())
              res = gcc_jit_type_get_volatile(res);
            return res;
          })
          .Case([&](gccjit::IntType t) {
            auto kind = t.getKind();
            return gcc_jit_context_get_type(ctxt, kind);
          })
          .Case([&](mlir::IntegerType t) {
            assert(!t.isSignless() && "signless integer type is not supported");
            return gcc_jit_context_get_int_type(ctxt, t.getWidth() / CHAR_BIT,
                                                t.isSigned());
          })
          .Case([&](mlir::IndexType) {
            return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_SIZE_T);
          })
          .Case([&](gccjit::FloatType t) {
            auto kind = t.getKind();
            return gcc_jit_context_get_type(ctxt, kind);
          })
          .Case([&](mlir::Float32Type) {
            return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_FLOAT);
          })
          .Case([&](mlir::Float64Type) {
            return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_DOUBLE);
          })
          .Case([&](gccjit::ComplexType t) {
            auto kind = t.getKind();
            return gcc_jit_context_get_type(ctxt, kind);
          })
          .Case([&](mlir::ComplexType t) {
            mlir::Type elementTy = t.getElementType();
            if (mlir::isa<mlir::Float32Type>(elementTy))
              return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_COMPLEX_FLOAT);
            if (mlir::isa<mlir::Float64Type>(elementTy))
              return gcc_jit_context_get_type(ctxt,
                                              GCC_JIT_TYPE_COMPLEX_DOUBLE);
            if (auto floatTy = mlir::dyn_cast<gccjit::FloatType>(elementTy)) {
              switch (floatTy.getKind()) {
              case GCC_JIT_TYPE_FLOAT:
                return gcc_jit_context_get_type(ctxt,
                                                GCC_JIT_TYPE_COMPLEX_FLOAT);
              case GCC_JIT_TYPE_DOUBLE:
                return gcc_jit_context_get_type(ctxt,
                                                GCC_JIT_TYPE_COMPLEX_DOUBLE);
              default:
                break;
              }
            }
            llvm_unreachable("unsupported complex element type");
          })
          .Case([&](gccjit::VoidType t) {
            return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_VOID);
          })
          .Case([&](gccjit::ArrayType t) {
            auto *elemType = convertType(t.getElementType());
            auto size = t.getSize();
            return gcc_jit_context_new_array_type(ctxt, nullptr, elemType,
                                                  size);
          })
          .Case([&](gccjit::VectorType t) {
            auto *elemType = convertType(t.getElementType());
            auto size = t.getNumUnits();
            return gcc_jit_type_get_vector(elemType, size);
          })
          .Case([&](mlir::VectorType t) {
            assert(!t.isScalable() &&
                   "scalable vector types are not supported");
            auto *elemType = convertType(t.getElementType());
            auto size = t.getNumElements();
            return gcc_jit_type_get_vector(elemType, size);
          })
          .Case([&](gccjit::StructType t) -> gcc_jit_type * {
            gcc_jit_struct *rawType = getOrCreateStructEntry(t).getRawHandle();
            return gcc_jit_struct_as_type(rawType);
          })
          .Case([&](gccjit::UnionType t) -> gcc_jit_type * {
            return getOrCreateUnionEntry(t).getRawHandle();
          })
          .Default([](mlir::Type) -> gcc_jit_type * {
            llvm_unreachable("unsupported type for gccjit translation");
          });
  typeMap[type] = res;
  return res;
}

template <typename RawCreator>
static auto
convertRecordType(GCCJITTranslation &translation,
                  GCCJITRecordTypeInterface type,
                  llvm::SmallVector<gcc_jit_field *> &convertedFields,
                  RawCreator &&rawHandleCreator) {
  static_assert(
      std::is_invocable_v<RawCreator &&, gcc_jit_context *, gcc_jit_location *,
                          const char *, int, gcc_jit_field **>);

  // TODO: handle opaque struct type.

  convertedFields.clear();
  convertedFields.reserve(type.getRecordFields().size());
  for (Attribute fieldOpaqueAttr : type.getRecordFields()) {
    auto fieldAttr = cast<FieldAttr>(fieldOpaqueAttr);

    int fieldBitWidth = fieldAttr.getBitWidth().value_or(0);
    std::string fieldName = fieldAttr.getName().str();
    gcc_jit_type *fieldType = translation.convertType(fieldAttr.getType());

    gcc_jit_location *loc = nullptr;
    if (auto fieldLoc = fieldAttr.getLoc())
      loc = translation.convertLocation(*fieldLoc);

    gcc_jit_field *field =
        fieldAttr.getBitWidth()
            ? gcc_jit_context_new_bitfield(translation.getContext(), loc,
                                           fieldType, fieldBitWidth,
                                           fieldName.c_str())
            : gcc_jit_context_new_field(translation.getContext(), loc,
                                        fieldType, fieldName.c_str());
    convertedFields.push_back(field);
  }

  std::string recordName = type.getRecordName().str();

  gcc_jit_location *loc = nullptr;
  if (SourceLocAttr recordLoc = type.getRecordLoc())
    loc = translation.convertLocation(recordLoc);

  return std::invoke(std::forward<RawCreator>(rawHandleCreator),
                     translation.getContext(), loc, recordName.c_str(),
                     convertedFields.size(), convertedFields.data());
}

GCCJITTranslation::StructEntry &
GCCJITTranslation::getOrCreateStructEntry(StructType type) {
  auto structMapIter = structMap.find(type);
  if (structMapIter == structMap.end()) {
    llvm::SmallVector<gcc_jit_field *> convertedFields;
    gcc_jit_struct *rawType = convertRecordType(
        *this, type, convertedFields, gcc_jit_context_new_struct_type);
    structMapIter = structMap.insert({type, StructEntry(rawType)}).first;
  }

  return structMapIter->second;
}

GCCJITTranslation::UnionEntry &
GCCJITTranslation::getOrCreateUnionEntry(UnionType type) {
  auto unionMapIter = unionMap.find(type);
  if (unionMapIter == unionMap.end()) {
    llvm::SmallVector<gcc_jit_field *> convertedFields;
    gcc_jit_type *rawType = convertRecordType(*this, type, convertedFields,
                                              gcc_jit_context_new_union_type);
    unionMapIter =
        unionMap.insert({type, UnionEntry(rawType, std::move(convertedFields))})
            .first;
  }

  return unionMapIter->second;
}

} // namespace mlir::gccjit
