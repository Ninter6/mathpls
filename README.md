# mathpls
You can think of it as a math library that students make up with notes.
It's to simple that I don't think anyone will use it.
But I will keep writing notes.

# Usage
This an easy-to-use header-only math library.
It covers many usage classes and functions, including vector, matrix and quaternion classes and also transformation functions like translate, rotate, scale and many useful function.

Its class and functions are all built on templates.

For example, you can set a vector with a custom number of elements like this.
```c++
using MyVec = mathpls::vec<float, 5>;
```
You can also easly define a matrix with the help of that.
```c++
mathpls::mat4 myMat4 = {
    mathpls::vec4{1, 2, 3, 4},
    mathpls::vec4{2, 4, 1, 3},
    mathpls::vec4{4, 3, 2, 1},
    mathpls::vec4{1, 4, 2, 3}
};
```
