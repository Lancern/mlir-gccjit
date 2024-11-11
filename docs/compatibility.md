# Compatibility

## Minimum GCCJIT ABI requirement

`mlir-gccjit` assumes the availability of [`LIBGCCJIT_ABI_27`](https://gcc.gnu.org/onlinedocs/jit/topics/compatibility.html#libgccjit-abi-27), which is available in GCC `14.2.0` or higher.

Please refer to [ABI and API compatibility](https://gcc.gnu.org/onlinedocs/jit/topics/compatibility.html#abi-and-api-compatibility) for detailed ABI specification of `libgccjit.so`.

## Known Issues

### Polyfill of `gccjit.alignof` 

[`LIBGCCJIT_ABI_28`](https://gcc.gnu.org/onlinedocs/jit/topics/compatibility.html#libgccjit-abi-28) introduces `gcc_jit_context_new_alignof`. On targets before such ABI level, we calculate the alignment of a given type `T` by creating
a dummy structure:
```c
struct dummy {
    char __padding;
    T __field;
};
```

The alignment is then calculate as the offset to `__field` if a `dummy` struct is placed at address `NULL`:
```c
bitcast(&((dummy *)NULL)->__field, size_t)
```

### GCCJIT ICE with Aligned Alloca or Scoped Alloca

When alignment is specified, our lowering pass converts `memref.alloca` to an intrinsic call to `__builtin_alloca_with_align`. Unfortunately, due to some implementation details inside GCC, `libgccjit` currently cannot identidy the type of this builtin function hence would run into an ICE. We have reported this project to upstream and will keep track of the status. Such issues also apply to
`__builtin_stack_save` and `__builtin_stack_restore`, which are required to implement scoped alloca.

