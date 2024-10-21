#ifndef GCCJIT_OPS
#define GCCJIT_OPS

include "mlir-gccjit/IR/GCCJITTypes.td"
include "mlir-gccjit/IR/GCCJITAttrs.td"

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/MemorySlotInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FNKIND_EXPORTED : I32EnumAttrCase<"Exported", 1, "exported">;
def FNKIND_INTERNAL : I32EnumAttrCase<"Internal", 2, "internal">;
def FNKIND_IMPORTED : I32EnumAttrCase<"Imported", 3, "imported">;
def FNKIND_ALWAYS_INLINE : I32EnumAttrCase<"AlwaysInline", 4, "always_inline">;

def FnKind : I32EnumAttr<"FnKind", "function kind", [
  FNKIND_EXPORTED, FNKIND_INTERNAL, FNKIND_IMPORTED, FNKIND_ALWAYS_INLINE
]> {
  let cppNamespace = "::mlir::gccjit";
}

def FuncOp : GCCJIT_Op<"func", [
  AutomaticAllocationScope, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove
]> {
    let summary = "Declare or define a function for GCCJIT";
    let description = [{
      FuncOp is the top-level operation for defining a new function in GCCJIT.
      Unlike func.func or llvm.func, parameters of a function are treated as
      lvalues in GCCJIT. Hence, `gccjit.func` only accepts regions with lvalue
      arguments if any parameter is present.
      
      ```
      gccjit.func @foo (i32, i32) -> i32 {
        ^entry(%arg0 : !gccjit.lvalue<i32>, %arg1 : !gccjit.lvalue<i32>):
          gccjit.return %2 : i32
      }
      gccjit.func always_inline @bar () -> void {
        ^entry:
          gccjit.return
      }
      ```

      `gccjit.func` can be of four kinds:
        - Exported: The function is visible to the outside world.
        - Internal: The function is visible only within the module, similar to
          `static` in C.
        - Imported: The function is imported from another module.
        - AlwaysInline: The function is always inlined.
    }];
    let arguments = (ins 
    SymbolNameAttr:$symName, 
        DefaultValuedAttr<FnKind, "FnKind::Exported">:$fnKind,
        TypeAttrOf<GCCJIT_FuncType>:$functionType,
        ArrayAttr:$gccjitFnAttrs,
        OptionalAttr<DictArrayAttr>:$argAttrs,
        OptionalAttr<DictArrayAttr>:$resAttrs
    );
    let regions = (region AnyRegion:$body);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        ::mlir::Region *getCallableRegion();

        /// Returns the results types that the callable region produces when
        /// executed.
        ArrayRef<Type> getCallableResults() {
            if (::llvm::isa<::mlir::gccjit::VoidType>(getFunctionType().getReturnType()))
                return {};
            return getFunctionType().getReturnTypes();
        }

        /// Returns the argument attributes for all callable region arguments or
        /// null if there are none.
        ::mlir::ArrayAttr getCallableArgAttrs() {
            return getArgAttrs().value_or(nullptr);
        }

        /// Returns the result attributes for all callable region results or null if
        /// there are none.
        ::mlir::ArrayAttr getCallableResAttrs() {
            return getResAttrs().value_or(nullptr);
        }

        /// Returns the argument types of this function.
        ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

        /// Returns the result types of this function.
        ArrayRef<Type> getResultTypes() { return getFunctionType().getReturnTypes(); }

        /// Hook for OpTrait::FunctionOpInterfaceTrait, called after verifying that
        /// the 'type' attribute is present and checks if it holds a function type.
        /// Ensures getType, getNumFuncArguments, and getNumFuncResults can be
        /// called safely.
        LogicalResult verifyType();

        //===------------------------------------------------------------------===//
        // SymbolOpInterface Methods
        //===------------------------------------------------------------------===//

        bool isDeclaration();

        bool hasAvailableExternallyLinkage() {
            return getFnKind() == FnKind::Exported;
        }
    }];
}

#endif // GCCJIT_OPS
