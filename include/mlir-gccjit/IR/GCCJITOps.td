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

def FuncOp : GCCJIT_Op<"func", [IsolatedFromAbove]> {
  let summary = "Declare or define a function for GCCJIT";
  let description = [{
    FuncOp is the top-level operation for defining a new function in GCCJIT.
    Unlike func.func or llvm.func, parameters of a function are treated as
    lvalues in GCCJIT. Hence, `gccjit.func` only accepts regions with lvalue
    arguments if any parameter is present.

    ```
    gccjit.func exported @foo (i32, i32) -> i32 {
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
      SymbolNameAttr:$sym_name,
      FnKind:$fn_kind,
      TypeAttrOf<GCCJIT_FuncType>:$function_type,
      ArrayAttr:$gccjit_fn_attrs,
      OptionalAttr<DictArrayAttr>:$arg_attrs,
      OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;

  let assemblyFormat = [{
    custom<FunctionKind>($fn_kind)
    $sym_name
    custom<FunctionType>($function_type)
    custom<FunctionAttrs>($gccjit_fn_attrs)
    (`arg_attrs` `(` $arg_attrs^ `)`)?
    (`res_attrs` `(` $res_attrs^ `)`)?
    custom<FunctionBody>($body) attr-dict
  }];
  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getReturnTypes(); }

    /// Returns nymber of arguments to the function.
    unsigned getNumArguments() { return getFunctionType().getNumInputs(); }

    /// Checks if the function is imported.
    bool isImported() { return getFnKind() == FnKind::Imported; }

    /// Get aliasee if the function is an alias.
    FlatSymbolRefAttr getAliasee();
  }];
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
def ReturnOp : GCCJIT_Op<"return", [Terminator, ReturnLike, ParentOneOf<["FuncOp"]>]> {
  let summary = "Return from function";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation.

    ```mlir
    gccjit.func @foo() {
      ^entry:
        gccjit.return
    }
    ```
  }];
  let arguments = (ins Optional<AnyType>:$value);
  let assemblyFormat = [{
    ($value^ `:` type($value))? attr-dict
  }];
  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasReturnValue() { return getNumOperands() != 0; }
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
def ConstantOp : GCCJIT_Op<"const", [ConstantLike, Pure,
                                     AllTypesMatch<["value", "result"]>]> {
  let summary = "A constant value";
  let description = [{
    The `gccjit.const` operation creates a constant rvalue for the constant
    value specified by the parameter.
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$value attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// JumpOp
//===----------------------------------------------------------------------===//
def JumpOp : GCCJIT_Op<"jump", [
  Terminator,
  Pure,
  ParentOneOf<["FuncOp"]>
]> {
  let summary = "Unconditional jump";
  let description = [{
    Terminate a block by adding a jump to the given target block.
    This is roughly equivalent to this C code:
    ```
    goto target;
    ```
    Unlike other branch op in MLIR, this jump op does not support any block arguments.
  }];
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ConditionalOp
//===----------------------------------------------------------------------===//
def ConditionalOp : GCCJIT_Op<"conditional", [
  Terminator,
  Pure,
  ParentOneOf<["FuncOp"]>
]> {
  let summary = "Unconditional jump";
  let description = [{
    Terminate a block by adding evaluation of an rvalue, branching on the result to the appropriate successor block.
    This is roughly equivalent to this C code:
    ```
    if (boolval)
      goto on_true;
    else
      goto on_false;
    ```
    Unlike other branch op in MLIR, this conditional op does not support any block arguments.
  }];
  let arguments = (ins GCCJIT_BoolType:$condition);
  let successors = (successor AnySuccessor:$on_true, AnySuccessor:$on_false);
  let assemblyFormat = [{
    `(` $condition `:` type($condition) `)` `,` $on_true `,` $on_false attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AsRValueOp
//===----------------------------------------------------------------------===//
def AsRValueOp : GCCJIT_Op<"as_rvalue"> {
  let summary = "Read a lvalue to rvalue";
  let description = [{
    The "as_rvalue" operation converts an lvalue to an rvalue.
    The operation takes a single operand and produces a single result.
  }];
  let arguments = (ins GCCJIT_LValueType:$lvalue);
  let results = (outs AnyType:$rvalue);
  let assemblyFormat = [{
      $lvalue `:` type($lvalue) `to` type($rvalue) attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//
def SwitchOp : GCCJIT_Op<"switch", [Terminator, Pure, ParentOneOf<["FuncOp"]>]> {
  let summary = "Switch operation";
  let description = [{
    The "switch" operation represents a switch operation within a block.

    ```mlir
    gccjit.func @foo() {
      ^entry:
        gccjit.switch (%0 : i32) {
          default: ^default
          case 0..1: ^case1
          case 5: ^case2
        }
      ^case1:
        gccjit.return
      ^case2:
        gccjit.return
      ^default:
    }
    ```
  }];
  let arguments = (
    ins GCCJIT_IntType:$value,
    ArrayAttr:$case_lowerbound,
    ArrayAttr:$case_upperbound
  );
  let successors = (successor
    AnySuccessor:$defaultDestination,
    VariadicSuccessor<AnySuccessor>:$caseDestinations
  );
  let assemblyFormat = [{
    `(` $value `:` type($value) `)` `{` `\n`
      custom<SwitchOpCases>(ref(type($value)),
                            $defaultDestination,
                            $case_lowerbound,
                            $case_upperbound,
                            $caseDestinations)
    `}`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// EvalOp
//===----------------------------------------------------------------------===//
def EvalOp : GCCJIT_Op<"eval"> {
  let summary = "Evaluate an expression";
  let description = [{
    The "eval" operation evaluates an expression.
    RValue expressions are lazy in GCCJIT. That is, if there is no side effect
    associated with an expression, related gimple is never constructed. One can
    force the evaluation to happen inside a block by using the "eval" operation.

    Roughtly, this is equivalent to the following C code:
    ```c
    (void)expr;
    ```
  }];
  let arguments = (ins AnyType:$expr);
  let assemblyFormat = [{
    `(` $expr `:` type($expr) `)` attr-dict
  }];
  let hasVerifier = 1;
}

#endif // GCCJIT_OPS
