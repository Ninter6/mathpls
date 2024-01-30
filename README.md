<!--
 * @Author: Ninter6 mc525740@outlook.com
 * @Date: 2024-01-31 00:41:52
 * @LastEditors: Ninter6
 * @LastEditTime: 2024-01-31 01:11:50
-->
# mathpls
You can think of it as a math library that students make up with notes.
It's to simple that I don't think anyone will use it.
But I will keep writing notes.

# Usage
This is an easy-to-use header-only math library.
It covers many usage classes and functions, including vector, matrix and quaternion classes and also transformation functions like translate, rotate, scale and many useful functions.

## Classes
Many useful template classes are provided.
- vec *vector*
- mat *matrix*
- qua *quaternion*
- ...

### vec
```c++
// vec template
using MyVec = mathpls::vec<float, 5>;

// vec declaration
mathpls::vec3 v3 = {1.f, 0, 0}; // vec defaultly ues float as data type
mathpls::vec3 v3_1 = {1.f}; // {1, 1, 1}

mathpls::dvec2 dv2 = {2., 3.}; // double vec
mathpls::ivec4 iv4 = {1, 2, 3, 4} // int vec

// normalize vec
auto nv3_1 = mathpls::normalize(v3); // 1, return a normalized vec
auto nv3_2 = v3.normalized(); // 2, the same as 1
v3.normal() // 3, normalize itself

// length
auto len = v3.length();
```

### mat
```c++
// mat template
using MyMat = mathpls::mat<float, 2, 3>; // W2H3 matrix

// mat declaration
mat3 mi{}; // identity matrix
mat3 m3_2{2.f} // {[2, 0, 0], [0, 2, 0], [0, 0, 2]}
mat3 m{
    vec3{},
    vec3{1.f},
    vec3{1.f, 1.f, 4.f, 5.f}
};
```

## Random
Mathpls also provides a simple random system

```c++
// random engine
mathpls::mt19937 emt{114};
mathpls::rand_sequence esq{514};

// random vec
vec3 rv3 = rand_vec3();
auto rMyVec = rand_vec<float, 5>();
```
