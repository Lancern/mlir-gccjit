
#include <stdint.h>
#include <stdio.h>

void print(int32_t x) {
    printf("%d\n", x);
}

int32_t read() {
    int32_t x;
    scanf("%d", &x);
    return x;
}
