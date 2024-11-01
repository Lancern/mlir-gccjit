#include "libgccjit.h"
#include "mlir-gccjit/Translation/TranslateToGCCJIT.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::gccjit {
void GCCJITTranslation::convertTypes(
    mlir::TypeRange types, llvm::SmallVector<gcc_jit_type *> &result) {
  for (auto type : types)
    result.push_back(convertType(type));
}

gcc_jit_type *GCCJITTranslation::convertType(mlir::Type type) {
  if (auto it = typeMap.find(type); it != typeMap.end())
    return it->second;
  auto *res = llvm::TypeSwitch<mlir::Type, gcc_jit_type *>(type)
                  .Case([&](gccjit::LValueType t) {
                    return convertType(t.getInnerType());
                  })
                  .Case([&](gccjit::PointerType t) {
                    auto *pointee = convertType(t.getElementType());
                    return gcc_jit_type_get_pointer(pointee);
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
                  .Case([&](gccjit::FloatType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(ctxt, kind);
                  })
                  .Case([&](gccjit::ComplexType t) {
                    auto kind = t.getKind();
                    return gcc_jit_context_get_type(ctxt, kind);
                  })
                  .Case([&](gccjit::VoidType t) {
                    return gcc_jit_context_get_type(ctxt, GCC_JIT_TYPE_VOID);
                  })
                  .Case([&](gccjit::ArrayType t) {
                    auto *elemType = convertType(t.getElementType());
                    auto size = t.getSize();
                    return gcc_jit_context_new_array_type(ctxt, nullptr,
                                                          elemType, size);
                  })
                  .Case([&](gccjit::VectorType t) {
                    auto *elemType = convertType(t.getElementType());
                    auto size = t.getNumUnits();
                    return gcc_jit_type_get_vector(elemType, size);
                  })
                  .Case([&](gccjit::StructType t) -> gcc_jit_type * {
                    gcc_jit_struct *rawType =
                        getOrCreateStructEntry(t).getRawHandle();
                    return gcc_jit_struct_as_type(rawType);
                  })
                  .Case([&](gccjit::UnionType t) -> gcc_jit_type * {
                    return getOrCreateUnionEntry(t).getRawHandle();
                  })
                  .Default([](mlir::Type) { return nullptr; });
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
  // TODO: encode location information in the record fields.

  convertedFields.clear();
  convertedFields.reserve(type.getRecordFields().size());
  for (Attribute fieldOpaqueAttr : type.getRecordFields()) {
    auto fieldAttr = cast<FieldAttr>(fieldOpaqueAttr);

    int fieldBitWidth = fieldAttr.getBitWidth();
    std::string fieldName = fieldAttr.getName().str();
    gcc_jit_type *fieldType = translation.convertType(fieldAttr.getType());

    gcc_jit_field *field =
        fieldAttr.getBitWidth()
            ? gcc_jit_context_new_bitfield(translation.getContext(),
                                           /*loc=*/nullptr, fieldType,
                                           fieldBitWidth, fieldName.c_str())
            : gcc_jit_context_new_field(translation.getContext(),
                                        /*loc=*/nullptr, fieldType,
                                        fieldName.c_str());
    convertedFields.push_back(field);
  }

  std::string recordName = type.getRecordName().str();

  gcc_jit_location *recordLoc = nullptr;
  if (type.getRecordLoc())
    recordLoc = translation.getLocation(type.getRecordLoc());

  return std::invoke(std::forward<RawCreator>(rawHandleCreator),
                     translation.getContext(), recordLoc, recordName.c_str(),
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
