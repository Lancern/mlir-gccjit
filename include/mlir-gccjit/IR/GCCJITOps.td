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
      ArrayAttr:$gccjit_fn_attrs
  );
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;

  let assemblyFormat = [{
    $fn_kind
    $sym_name
    custom<FunctionType>($function_type)
    custom<FunctionAttrs>($gccjit_fn_attrs)
    custom<FunctionBody>($body) attr-dict
  }];
  let extraClassDeclaration = [{
    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() {
      return getFunctionType().getReturnTypes();
    }

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
def ReturnOp : GCCJIT_Op<"return",
    [Terminator, ReturnLike, ParentOneOf<["FuncOp", "GlobalOp", "ExprOp"]>]> {
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
    Unlike other branch op in MLIR, this jump op does not support any block
    arguments.
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
    Terminate a block by adding evaluation of an rvalue, branching on the result
    to the appropriate successor block. This is roughly equivalent to this C
    code:

    ```
    if (boolval)
      goto on_true;
    else
      goto on_false;
    ```

    Unlike other branch op in MLIR, this conditional op does not support any
    block arguments.
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
def SwitchOp : GCCJIT_Op<"switch",
    [Terminator, Pure, ParentOneOf<["FuncOp"]>]> {
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
// LocalOp
//===----------------------------------------------------------------------===//
def LocalOp : GCCJIT_Op<"local"> {
  let summary = "Declare a local variable";
  let description = [{
    The "local_var" operation declares a local variable.
  }];
  let arguments = (ins 
    OptionalAttr<StrAttr>:$reg_name,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<TLSModelAttr>:$tls_model
  );
  let results = (outs GCCJIT_LValueType:$var);
  let assemblyFormat = [{
    oilist ( 
      `reg` `(` qualified($reg_name) `)` |
      `align` `(` qualified($alignment) `)` |
      `tls_model` `(` qualified($tls_model) `)`
    )
    `:` type($var) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// GlobalOp
//===----------------------------------------------------------------------===//
def GLBKIND_EXPORTED : I32EnumAttrCase<"Exported", 0, "exported">;
def GLBKIND_INTERNAL : I32EnumAttrCase<"Internal", 1, "internal">;
def GLBKIND_IMPORTED : I32EnumAttrCase<"Imported", 2, "imported">;

def GlbKind : I32EnumAttr<"GlbKind", "global kind", [
  GLBKIND_EXPORTED, GLBKIND_INTERNAL, GLBKIND_IMPORTED
]> {
  let cppNamespace = "::mlir::gccjit";
}

def GlobalOp : GCCJIT_Op<"global", [IsolatedFromAbove]> {
  let summary = "Declare a global variable";
  let description = [{
    The "global_var" operation declares a global variable.
  }];
  let arguments = (ins 
    GlbKind:$glb_kind,
    SymbolNameAttr:$sym_name,
    TypeAttrOf<GCCJIT_LValueType>:$type,
    OptionalAttr<StrAttr>:$reg_name,
    OptionalAttr<IndexAttr>:$alignment,
    OptionalAttr<TLSModelAttr>:$tls_model,
    OptionalAttr<StrAttr>:$link_section,
    OptionalAttr<StrAttr>:$visibility,
    OptionalAttr<AnyAttrOf<[StrAttr, ByteArrayInitializerAttr]>>:$initializer
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    $glb_kind
    $sym_name
    oilist ( 
      `reg` `(` qualified($reg_name) `)` |
      `align` `(` qualified($alignment) `)` |
      `tls_model` `(` qualified($tls_model) `)` |
      `link_section` `(` qualified($link_section) `)` |
      `visibility` `(` qualified($visibility) `)`
    )
    custom<GlobalInitializer>($initializer, $body) `:` $type attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// RValue Expressions
//===----------------------------------------------------------------------===//

def ExprOp : GCCJIT_Op<"expr"> {
  let summary = "Expression";
  let description = [{
    The "expr" operation represents an expression.
    ```mlir
    %res = gccjit.expr {
      %0 = gccjit.const #gccjit.int<32> : !gccjit.int<32>
      %1 = gccjit.const #gccjit.int<32> : !gccjit.int<32>
      %2 = gccjit.binary plus %0, %1 : !gccjit.int<32>, !gccjit.int<32>
      gccjit.return %2 : !gccjit.int<32>
    } : !gccjit.int<32>
    The expr operation can be marked with a lazy attribute. If such an attribute
    exists, the expr will not be materialized until it is used.
    ```
  }];
  let arguments = (ins UnitAttr:$lazy);
  let results = (outs AnyType:$result);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    custom<LazyAttribute>($lazy) $body `:` type($result) attr-dict
  }];
}


//===----------------------------------------------------------------------===//
// Value construction operations.
//===----------------------------------------------------------------------===//

def SizeOfOp : GCCJIT_Op<"sizeof", [Pure]> {
  let summary = "Size of a type";
  let description = [{
    The "sizeof" operation returns the size of a type in bytes.
    ```mlir
    %size = gccjit.sizeof !gccjit.int<long double> : !gccjit.int<size_t>
    ```
  }];
  let arguments = (ins TypeAttr:$type);
  let results = (outs GCCJIT_IntType:$size);
  let assemblyFormat = [{
    $type `:` type($size) attr-dict
  }];
}

def AlignOfOp : GCCJIT_Op<"alignof", [Pure]> {
  let summary = "Alignment of a type";
  let description = [{
    The "alignof" operation returns the alignment of a type in bytes.
    ```mlir
    %align = gccjit.alignof !gccjit.int<long double> : !gccjit.int<size_t>
    ```
  }];
  let arguments = (ins TypeAttr:$type);
  let results = (outs GCCJIT_IntType:$align);
  let assemblyFormat = [{
    $type `:` type($align) attr-dict
  }];
}

def NewArrayOp : GCCJIT_Op<"new_array"> {
  let summary = "Construct an array";
  let description = [{
    The "array" operation constructs an array from a list of elements.
    ```mlir
    %array = gccjit.new_array !gccjit.array<i32, 3> [%1, %2, %3]
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs GCCJIT_ArrayType:$array);
  let assemblyFormat = [{
    type($array) `[`
      custom<ArrayOrVectorElements>(
        ref(type($array)), $elements, type($elements)
      )
    `]` attr-dict
  }];
}

def NewStructOp : GCCJIT_Op<"new_struct"> {
  let summary = "Construct a struct";
  let description = [{
    The "struct" operation constructs a struct from a list of elements.
    Each value has to have the same unqualified type as the field it is applied
    to.

    ```mlir
    %0 = gccjit.new_struct [0, 1] [%1, %2] : (!i32, !i32) -> !gccjit.struct<(!i32, !i32)>
    ```
  }];
  let arguments = (ins 
    DenseI32ArrayAttr:$indices,
    Variadic<AnyType>:$elements
  );
  // TODO: Add support for struct type.
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
     $indices `[` $elements `]` `:` functional-type(operands, results) attr-dict
  }];
}

def NewUnionOp : GCCJIT_Op<"new_union"> {
  let summary = "Construct a union";
  let description = [{
    The "union" operation constructs a union from a list of elements.
    Each value has to have the same unqualified type as the field it is applied
    to.
    ```mlir
    %union = gccjit.new_union %1 at 0 : !i32, !gccjit.union<(!i32)> 
    ```
  }];
  let arguments = (ins 
    AnyType:$element,
    IndexAttr:$index
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
     $element `at` $index `:` type($element) `,` type($result) attr-dict
  }];
}

def NewVectorOp : GCCJIT_Op<"new_vector"> {
  let summary = "Construct a vector";
  let description = [{
    The "vector" operation constructs a vector from a list of elements.
    ```mlir
    %vector = gccjit.new_vector !gccjit.vector<i32, 4> [%1, %2, %3, %4]
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs GCCJIT_VectorType:$vector);
  let assemblyFormat = [{
    type($vector) `[`
      custom<ArrayOrVectorElements>(
        ref(type($vector)),
        $elements,
        type($elements)
      )
    `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

def UOp_Minus : I32EnumAttrCase<"Minus", 0, "minus">;
def UOp_BitwiseNegate : I32EnumAttrCase<"BitwiseNegate", 1, "bitwise_negate">;
def UOp_LogicalNegate : I32EnumAttrCase<"LogicalNegate", 2, "logical_negate">;
def UOp_Abs : I32EnumAttrCase<"Abs", 3, "abs">;
def UOpAttr : I32EnumAttr<"UOp", "unary operation", [
  UOp_Minus, UOp_BitwiseNegate, UOp_LogicalNegate, UOp_Abs
]> {
  let cppNamespace = "::mlir::gccjit";
}

def UnaryOp : GCCJIT_Op<"unary"> {
  let summary = "Unary operation";
  let description = [{
    The "unary" operation represents a unary operation.
    ```mlir
    %res = gccjit.unary minus ( %operand : !gccjit.int<i32> ) : !gccjit.int<i32>
    ```
  }];
  let arguments = (ins UOpAttr:$op, AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $op `(` $value `:` type($value) `)` `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

def BOp_Plus : I32EnumAttrCase<"Plus", 0, "plus">;
def BOp_Minus : I32EnumAttrCase<"Minus", 1, "minus">;
def BOp_Mult : I32EnumAttrCase<"Mult", 2, "mult">;
def BOp_Divide : I32EnumAttrCase<"Divide", 3, "divide">;
def BOp_Modulo : I32EnumAttrCase<"Modulo", 4, "modulo">;
def BOp_BitwiseAnd : I32EnumAttrCase<"BitwiseAnd", 5, "bitwise_and">;
def BOp_BitwiseOr : I32EnumAttrCase<"BitwiseXor", 6, "bitwise_xor">;
def BOp_BitwiseXor : I32EnumAttrCase<"BitwiseOr", 7, "bitwise_or">;
def BOp_LogicalAnd : I32EnumAttrCase<"LogicalAnd", 8, "logical_and">;
def BOp_LogicalOr : I32EnumAttrCase<"LogicalOr", 9, "logical_or">;
def BOp_LShift : I32EnumAttrCase<"LShift", 10, "lshift">;
def BOp_RShift : I32EnumAttrCase<"RShift", 11, "rshift">;

def BOpAttr : I32EnumAttr<"BOp", "binary operation", [
  BOp_Plus, BOp_Minus, BOp_Mult, BOp_Divide, BOp_Modulo,
  BOp_BitwiseAnd, BOp_BitwiseOr, BOp_BitwiseXor,
  BOp_LogicalAnd, BOp_LogicalOr, BOp_LShift, BOp_RShift
]> {
  let cppNamespace = "::mlir::gccjit";
}

def BinaryOp : GCCJIT_Op<"binary"> {
  let summary = "Binary operation";
  let description = [{
    The "binary" operation represents a binary operation.

    ```mlir
    %res = gccjit.binary plus ( %lhs : !gccjit.int<i32>, %rhs : !gccjit.int<i32> ) : !gccjit.int<i32>
    ```
  }];
  let arguments = (ins BOpAttr:$op, AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $op `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)`
    `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Comparison Operations
//===----------------------------------------------------------------------===//
def CmpOp_Eq : I32EnumAttrCase<"Eq", 0, "eq">;
def CmpOp_Ne : I32EnumAttrCase<"Ne", 1, "ne">;
def CmpOp_Lt : I32EnumAttrCase<"Lt", 2, "lt">;
def CmpOp_Le : I32EnumAttrCase<"Le", 3, "le">;
def CmpOp_Gt : I32EnumAttrCase<"Gt", 4, "gt">;
def CmpOp_Ge : I32EnumAttrCase<"Ge", 5, "ge">;

def CmpOpAttr : I32EnumAttr<"CmpOp", "comparison operation", [
  CmpOp_Eq, CmpOp_Ne, CmpOp_Lt, CmpOp_Le, CmpOp_Gt, CmpOp_Ge
]> {
  let cppNamespace = "::mlir::gccjit";
}

def CompareOp : GCCJIT_Op<"compare"> {
  let summary = "Comparison operation";
  let description = [{
    The "compare" operation represents a comparison operation.

    ```mlir
    %res = gccjit.compare eq ( %lhs : !gccjit.int<i32>, %rhs : !gccjit.int<i32> ) : !gccjit.bool
    ```
  }];
  let arguments = (ins CmpOpAttr:$op, AnyType:$lhs, AnyType:$rhs);
  let results = (outs GCCJIT_BoolType:$result);
  let assemblyFormat = [{
    $op `(` $lhs `:` type($lhs) `,` $rhs `:` type($rhs) `)`
    `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Function Call
//===----------------------------------------------------------------------===//
def CallOp : GCCJIT_Op<"call"> {
  let summary = "Function call";
  let description = [{
    The "call" operation represents a function call.

    ```mlir
    %res = gccjit.call @foo ( %arg0 : !gccjit.int<i32>, %arg1 : !gccjit.int<i32> ) : !gccjit.int<i32>
    ```

    If tail call is required, the `tail` attribute should be set to true.

    ```mlir
    %res = gccjit.call tail @foo ( %arg0 : !gccjit.int<i32>, %arg1 : !gccjit.int<i32> ) : !gccjit.int<i32>
    ```
  }];
  let arguments = (ins SymbolRefAttr:$callee, 
    Variadic<AnyType>:$args, 
    UnitAttr:$tail,
    UnitAttr:$builtin
  );
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    custom<TailCallAttr>($tail)
    custom<BuiltinCallAttr>($builtin)
    $callee `(` $args `)` `:` functional-type($args, $result) attr-dict
  }];
}

def PtrCallOp : GCCJIT_Op<"ptr_call"> {
  let summary = "Function pointer call";
  let description = [{
    The "ptr_call" operation represents a function pointer call.

    ```mlir
    %res = gccjit.ptr_call %fn ( %arg0 : !gccjit.int<i32>, %arg1 : !gccjit.int<i32> ) : !gccjit.int<i32>
    ```

    If tail call is required, the `tail` attribute should be set to true.

    ```mlir
    %res = gccjit.ptr_call tail %fn ( %arg0 : !gccjit.int<i32>, %arg1 : !gccjit.int<i32> ) : !gccjit.int<i32>
    ```
  }];
  let arguments = (ins GCCJIT_PointerType:$callee, Variadic<AnyType>:$args, UnitAttr:$tail);
  let results = (outs Optional<AnyType>:$result);
  let assemblyFormat = [{
    custom<TailCallAttr>($tail) 
    $callee `(` $args `)` `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Casting
//===----------------------------------------------------------------------===//
def CastOp : GCCJIT_Op<"cast"> {
  let summary = "Cast operation";
  let description = [{
    The "cast" operation represents a casting operation.

    ```mlir
    %res = gccjit.cast %value : !gccjit.int<i32> to !gccjit.int<i64>
    ```

    Currently only a limited set of conversions are possible:
    - `int <-> float`
    - `int <-> pointer`
    - `pointer <-> pointer`
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `:`  type($value) `to` type($result) attr-dict
  }];
}

def BitCastOp : GCCJIT_Op<"bitcast"> {
  let summary = "Bitcast operation";
  let description = [{
    Given an rvalue of T, bitcast it to another type, 
    meaning that this will generate a new rvalue by interpreting 
    the bits of rvalue to the layout of type.

    The type of rvalue must be the same size as the size of type.

    ```mlir
    %res = gccjit.bitcast %value : !gccjit.int<long> to !gccjit.int<unsigned long>
    ```
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $value `:`  type($value) `to` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// LiteralOp
//===----------------------------------------------------------------------===//
def LiteralOp : GCCJIT_Op<"literal"> {
  let summary = "Literal operation";
  let description = [{
    The "literal" operation represents a literal value.
    ```mlir
    !str = !gccjit.qualified<!gccjit.int<char>, const>
    %res = gccjit.literal #gccjit.str<"Hello, World!"> : !str
    ```
  }];
  let arguments = (ins StrAttr:$value);
  let results = (outs GCCJIT_PointerType:$result);
  let assemblyFormat = [{
    $value `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Addr Operation
//===----------------------------------------------------------------------===//
def AddrOp : GCCJIT_Op<"addr"> {
  let summary = "Address of an lvalue";
  let description = [{
    The "addr" operation returns the address of an lvalue.

    ```mlir
    %res = gccjit.addr (%lvalue : !gccjit.lvalue<!i32>) : !gccjit.ptr<!i32>
    ```
  }];
  let arguments = (ins GCCJIT_LValueType:$lvalue);
  let results = (outs GCCJIT_PointerType:$result);
  let assemblyFormat = [{
    `(` $lvalue `:` type($lvalue) `)` `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FnAddr Operation
//===----------------------------------------------------------------------===//
def FnAddrOp : GCCJIT_Op<"fn_addr"> {
  let summary = "Function address";
  let description = [{
    The "fn_addr" operation returns the address of a function.

    ```mlir
    %res = gccjit.fn_addr @foo : !gccjit.ptr<!gccjit.func<void (!i32)>>
    ```
  }];
  let arguments = (ins SymbolRefAttr:$callee);
  let results = (outs GCCJIT_PointerType:$result);
  let assemblyFormat = [{
    $callee `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Deref Operation
//===----------------------------------------------------------------------===//
def DerefOp : GCCJIT_Op<"deref"> {
  let summary = "Dereference a pointer";
  let description = [{
    The "deref" operation dereferences a pointer.

    ```mlir
    %res = gccjit.deref (%ptr : !gccjit.ptr<!i32>) : !gccjit.lvalue<!i32>
    ```
  }];
  let arguments = (ins GCCJIT_PointerType:$ptr, Optional<GCCJIT_IntType>:$offset);
  let results = (outs GCCJIT_LValueType:$result);
  let assemblyFormat = [{
    `(` $ptr `:` type($ptr) (`,` $offset^ `:` type($offset) )? `)`
    `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Assignment Operation
//===----------------------------------------------------------------------===//
def AssignOp : GCCJIT_Op<"assign"> {
  let summary = "Assignment operation";
  let description = [{
    The "assign" operation represents an assignment operation.

    ```mlir
    gccjit.assign %rvalue to $lvalue : !gccjit.lvalue<!i32>, !i32
    ```
  }];
  let arguments = (ins AnyType:$rvalue, GCCJIT_LValueType:$lvalue);
  let assemblyFormat = [{
    $rvalue `to` $lvalue `:` type($rvalue) `,` type($lvalue) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Update Operation
//===----------------------------------------------------------------------===//
def UpdateOp : GCCJIT_Op<"update"> {
  let summary = "Update operation";
  let description = [{
    The "update" operation represents an update operation.

    ```mlir
    gccjit.update minus %rvalue to $lvalue : !gccjit.lvalue<!i32>, !i32
    ```
  }];
  let arguments = (ins BOpAttr:$op, AnyType:$rvalue, GCCJIT_LValueType:$lvalue);
  let assemblyFormat = [{
    $op $rvalue `to` $lvalue `:` type($rvalue) `,` type($lvalue) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Comment Operation
//===----------------------------------------------------------------------===//
def CommentOp : GCCJIT_Op<"comment", [Pure]> {
  let summary = "Comment operation";
  let description = [{
    The "comment" operation represents a comment.

    ```mlir
    gccjit.comment #gccjit.str<"This is a comment">
    ```
  }];
  let arguments = (ins StrAttr:$comment);
  let assemblyFormat = [{
    $comment attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Access field
//===----------------------------------------------------------------------===//
def AccessFieldOp : GCCJIT_Op<"access_field"> {
  let summary = "Access field operation";
  let description = [{
    The "access_field" operation represents an access field operation.
    LValue field access returns an lvalue.
    RValue field access returns an rvalue.

    ```mlir
    %res0 = gccjit.access_field %struct [0] : !gccjit.lvalue<!gccjit.struct<(!i32, !i32)>> -> !gccjit.lvalue<!i32>
    %res1 = gccjit.access_field %struct [1] : !gccjit.struct<(!i32, !i32)> -> !i32
    ```
  }];
  let arguments = (ins AnyType:$composite, IndexAttr:$field);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $composite `[` $field `]` `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Deref Field
//===----------------------------------------------------------------------===//
def DerefFieldOp : GCCJIT_Op<"deref_field"> {
  let summary = "Dereference field operation";
  let description = [{
    The "deref_field" operation represents a dereference field operation.

    ```mlir
    %res = gccjit.deref_field %ptr [0] : !gccjit.ptr<!gccjit.struct<(!i32, !i32)>> -> !gccjit.lvalue<!i32>
    ```
  }];
  let arguments = (ins GCCJIT_PointerType:$ptr, IndexAttr:$field);
  let results = (outs GCCJIT_LValueType:$result);
  let assemblyFormat = [{
    $ptr `[` $field `]` `:` functional-type(operands, results) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// GetGlobalOp
//===----------------------------------------------------------------------===//
def GetGlobalOp : GCCJIT_Op<"get_global"> {
  let summary = "Get global operation";
  let description = [{
    The "get_global" operation is used to get a lvalue of a global variable.

    ```mlir
    %res = gccjit.get_global @foo : !gccjit.lvalue<!i32>
    ```
  }];
  let arguments = (ins SymbolRefAttr:$sym);
  let results = (outs GCCJIT_LValueType:$result);
  let assemblyFormat = [{
    $sym `:` type($result) attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AsmOp
//===----------------------------------------------------------------------===//
def AsmOp : GCCJIT_Op<"asm", [AttrSizedOperandSegments]> {
  let summary = "Inline assembly operation";
  let description = [{
    The "asm" operation represents an inline assembly operation.

    ```mlir
    gccjit.asm inline volatile ("..." 
      : "=r"(%result) 
      : "r"(%arg0), "r"(%arg1) 
      : "memory"
    )
    ```
  }];
  let arguments = (ins 
    UnitAttr:$inline_flag,
    UnitAttr:$volatile_flag,
    StrAttr:$template_code,
    ArrayAttr:$outputConstraints,
    ArrayAttr:$outputSymbols,
    Variadic<GCCJIT_LValueType>:$outputs,
    ArrayAttr:$inputConstraints,
    ArrayAttr:$inputSymbols,
    Variadic<AnyType>:$inputs,
    ArrayAttr:$clobbers
  );
  let assemblyFormat = [{
    custom<AsmInlineAttr>($inline_flag)
    custom<AsmVolatileAttr>($volatile_flag)
    `(` 
       $template_code
      `:` custom<AsmOperands>($outputConstraints, $outputSymbols, $outputs,
                              type($outputs))
      `:` custom<AsmOperands>($inputConstraints, $inputSymbols, $inputs,
                              type($inputs))
      `:` custom<ClobberList>($clobbers)
    `)` attr-dict
  }];
}

///===----------------------------------------------------------------------===//
/// AsmGotoOp
///===----------------------------------------------------------------------===//
def AsmGotoOp : GCCJIT_Op<"asm_goto",
    [Terminator, ParentOneOf<["FuncOp"]>, AttrSizedOperandSegments]> {
  let summary = "Inline assembly operation";
  let description = [{
    The "asm" operation represents an inline assembly operation.

    ```mlir
    gccjit.asm_goto inline volatile ("..." 
      : "=r"(%result) 
      : "r"(%arg0), "r"(%arg1) 
      : "memory"
      : /* goto labels */ label1, label2
      : /* fallthrough label */ "label3"
    )
    ```
  }];
  let arguments = (ins 
    UnitAttr:$inline_flag,
    UnitAttr:$volatile_flag,
    StrAttr:$template_code,
    ArrayAttr:$outputConstraints,
    ArrayAttr:$outputSymbols,
    Variadic<GCCJIT_LValueType>:$outputs,
    ArrayAttr:$inputConstraints,
    ArrayAttr:$inputSymbols,
    Variadic<AnyType>:$inputs,
    ArrayAttr:$clobbers
  );
  let successors = (successor AnySuccessor:$fallthrough,
                              VariadicSuccessor<AnySuccessor>:$labels);
  let assemblyFormat = [{
    custom<AsmInlineAttr>($inline_flag)
    custom<AsmVolatileAttr>($volatile_flag)
    `(` 
       $template_code
      `:` custom<AsmOperands>($outputConstraints, $outputSymbols, $outputs,
                              type($outputs))
      `:` custom<AsmOperands>($inputConstraints, $inputSymbols, $inputs,
                              type($inputs))
      `:` custom<ClobberList>($clobbers)
      `:` $labels
      `:` $fallthrough
    `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Type Generic Atomic Operations
//===----------------------------------------------------------------------===//
def ATOMIC_ORD_RELAXED : I32EnumAttrCase<"Relaxed", 0, "relaxed">;
def ATOMIC_ORD_CONSUME : I32EnumAttrCase<"Consume", 1, "consume">;
def ATOMIC_ORD_ACQUIRE : I32EnumAttrCase<"Acquire", 2, "acquire">;
def ATOMIC_ORD_RELEASE : I32EnumAttrCase<"Release", 3, "release">;
def ATOMIC_ORD_ACQ_REL : I32EnumAttrCase<"AcqRel", 4, "acq_rel">;
def ATOMIC_ORD_SEQ_CST : I32EnumAttrCase<"SeqCst", 5, "seq_cst">;

def AtomicOrderingAttr : I32EnumAttr<"AtomicOrdering", "atomic ordering", [
  ATOMIC_ORD_RELAXED, ATOMIC_ORD_CONSUME, ATOMIC_ORD_ACQUIRE,
  ATOMIC_ORD_RELEASE, ATOMIC_ORD_ACQ_REL, ATOMIC_ORD_SEQ_CST
]> {
  let cppNamespace = "::mlir::gccjit";
}

def ATOMIC_RMW_STORE : I32EnumAttrCase<"Store", 0, "store">;
def ATOMIC_RMW_XCHG : I32EnumAttrCase<"Xchg", 1, "xchg">;
def ATOMIC_RMW_ADD_FETCH : I32EnumAttrCase<"AddFetch", 2, "add_fetch">;
def ATOMIC_RMW_SUB_FETCH : I32EnumAttrCase<"SubFetch", 3, "sub_fetch">;
def ATOMIC_RMW_AND_FETCH : I32EnumAttrCase<"AndFetch", 4, "and_fetch">;
def ATOMIC_RMW_NAND_FETCH : I32EnumAttrCase<"NandFetch", 5, "nand_fetch">;
def ATOMIC_RMW_OR_FETCH : I32EnumAttrCase<"OrFetch", 6, "or_fetch">;
def ATOMIC_RMW_XOR_FETCH : I32EnumAttrCase<"XorFetch", 7, "xor_fetch">;
def ATOMIC_RMW_FETCH_ADD : I32EnumAttrCase<"FetchAdd", 8, "fetch_add">;
def ATOMIC_RMW_FETCH_SUB : I32EnumAttrCase<"FetchSub", 9, "fetch_sub">;
def ATOMIC_RMW_FETCH_AND : I32EnumAttrCase<"FetchAnd", 10, "fetch_and">;
def ATOMIC_RMW_FETCH_NAND : I32EnumAttrCase<"FetchNand", 11, "fetch_nand">;
def ATOMIC_RMW_FETCH_OR : I32EnumAttrCase<"FetchOr", 12, "fetch_or">;
def ATOMIC_RMW_FETCH_XOR : I32EnumAttrCase<"FetchXor", 13, "fetch_xor">;

def AtomicRMWKindAttr : I32EnumAttr<"AtomicRMWKind", "atomic rmw operation", [
  ATOMIC_RMW_STORE, ATOMIC_RMW_XCHG, ATOMIC_RMW_ADD_FETCH, ATOMIC_RMW_SUB_FETCH,
  ATOMIC_RMW_AND_FETCH, ATOMIC_RMW_NAND_FETCH, ATOMIC_RMW_OR_FETCH,
  ATOMIC_RMW_XOR_FETCH, ATOMIC_RMW_FETCH_ADD, ATOMIC_RMW_FETCH_SUB,
  ATOMIC_RMW_FETCH_AND, ATOMIC_RMW_FETCH_NAND, ATOMIC_RMW_FETCH_OR,
  ATOMIC_RMW_FETCH_XOR
]> {
  let cppNamespace = "::mlir::gccjit";
}

def AtomicLoadOp : GCCJIT_Op<"atomic.load"> {
  let summary = "Atomic load operation";
  let description = [{
    The `gccjit.atomic.load` operation represents an atomic load operation.

    ```mlir
    %res = gccjit.atomic.load relaxed (%ptr : !gccjit.ptr<!i32>) : !i32
    ```
  }];
  let arguments = (ins GCCJIT_PointerType:$ptr, AtomicOrderingAttr:$ordering);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ordering `(` $ptr `:` type($ptr) `)` `:` type($result) attr-dict
  }];
}

def AtomicRMWOp : GCCJIT_Op<"atomic.rmw"> {
  let summary = "Atomic rmw operation";
  let description = [{
    The `gccjit.atomic.rmw` operation represents an atomic rmw operation.

    ```mlir
    %res = gccjit.atomic.rmw relaxed add_fetch (%ptr : !gccjit.ptr<!i32>, %value : !i32) : !i32
    ```
  }];
  let arguments = (ins GCCJIT_PointerType:$ptr, AnyType:$value, AtomicRMWKindAttr:$kind, AtomicOrderingAttr:$ordering);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ordering $kind  `(` $ptr `:` type($ptr) `,` $value `:` type($value) `)`
    `:` type($result) attr-dict
  }];
}

def AtomicCompareExchangeOp : GCCJIT_Op<"atomic.cmpxchg"> {
  let summary = "Atomic compare exchange operation";
  let description = [{
    The `gccjit.atomic.cmpxchg` operation represents an atomic compare exchange operation.

    ```mlir
    %res = gccjit.atomic.cmpxchg weak success(acq_rel) failure(relaxed) 
      (%ptr : !gccjit.ptr<!i32>, %expected : !gccjit.ptr<!i32>, %desired : !i32) : !i1
    ```
  }];
  let arguments = (ins 
    GCCJIT_PointerType:$ptr, 
    GCCJIT_PointerType:$expected, 
    AnyType:$desired, 
    AtomicOrderingAttr:$success_ord, 
    AtomicOrderingAttr:$failure_ord, 
    UnitAttr:$weak);
  let results = (outs GCCJIT_BoolType:$result);
  let assemblyFormat = [{
    custom<WeakAttr>($weak)
    `success` `(` $success_ord `)` 
    `failure` `(` $failure_ord `)`
    `(`
      $ptr `:` type($ptr) `,`
      $expected `:` type($expected) `,`
      $desired `:` type($desired)
    `)` `:` type($result) attr-dict
  }];
}

#endif // GCCJIT_OPS
