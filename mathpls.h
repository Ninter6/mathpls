#pragma once

namespace mathpls {

namespace utils {

template <class T1, class T2>
constexpr bool is_same_v = false;

template <class T>
constexpr bool is_same_v<T, T> = true;

template <bool, class = void> class enable_if;
template <class T> class enable_if<false, T> {};

template <class T>
struct enable_if<true, T> {
    using type = T;
};

template <bool V, class T = void>
using enable_if_t = enable_if<V, T>::type;

template <class T>
struct remove_reference {
    using type = T;
};

template <class T>
struct remove_reference<T&> {
    using type = T;
};

template <class T>
struct remove_reference<T&&> {
    using type = T;
};

template <class T>
using remove_reference_t = remove_reference<T>::type;

template <typename T>
struct remove_cv {
    using type = T;
};

template <typename T>
struct remove_cv<const T> {
    using type = T;
};

template <typename T>
struct remove_cv<volatile T> {
    using type = T;
};

template <typename T>
struct remove_cv<const volatile T> {
    using type = T;
};

template <class T>
using remove_cv_t = remove_cv<T>::type;

template <class T>
using remove_cvref_t = remove_cv_t<remove_reference_t<T>>;

}

// useful tool functions

constexpr auto max(auto a, auto b) {
    return a>b ? a:b;
}

constexpr auto min(auto a, auto b) {
    return a<b ? a:b;
}

constexpr auto clamp(auto min, auto a, auto max) {
    return (min<(a<max?a:max)?(a<max?a:max):min<(max?a:max)?min:(max<a?a:max));
}

template <class T>
constexpr T max(T a, T b) {
    return a>b ? a:b;
}

template <class T>
constexpr T min(T a, T b) {
    return a<b ? a:b;
}

template <class T>
constexpr T clamp(T min, T a, T max) {
    return (min<(a<max?a:max)?(a<max?a:max):min<(max?a:max)?min:(max<a?a:max));
}

template <class T>
constexpr T abs(T a) {
    return a > 0 ? a : -a;
}

template <class T>
constexpr T pi() {return 3.14159265358979323846264338327950288;}
constexpr float pi() {return 3.14159265358979323846264338327950288;}

template<class T = float>
constexpr T radians(T angle) {
    return angle / T{180} * pi<T>();
}

template <class T, class Tt>
constexpr auto lerp(T a, T b, Tt t) {
    return a + (b - a) * t;
}

// following angle-related functions will ues this type
using angle_t = double;

constexpr angle_t sqrt(angle_t x) {
    if (x == 1 || x == 0)
        return x;
    double temp = x / 2;
    while (abs(temp - (temp + x / temp) / 2) > 1e-6)
        temp = (temp + x / temp) / 2;
    return temp;
}

constexpr angle_t pow(angle_t ori, angle_t a) {
    if(a < 0) return 1. / pow(ori, -a);
    int ip = a;
    angle_t fp = a - ip;
    angle_t r = 1;
    while(ip--) r *= ori;
    constexpr angle_t c[] = {0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 0.00006103515625, 0.000030517578125, 0.0000152587890625};
    angle_t t = ori;
    for(int i=0; fp >= c[15]; i++){
        t = sqrt(t);
        if(fp < c[i]) continue;
        fp -= c[i];
        r *= t;
    }
    return r;
}

// 三角函数这里对精度和性能上做了很多取舍,目前基本上已经是最理想的情况了,可以保证小数点后4位没有误差
constexpr angle_t sin(angle_t a) {
    if(a < 0) return -sin(-a); // sin(-a) = -sin(a)
    
    constexpr int
    angle[] = {23040, 13601, 7187, 3648, 1831, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1};
    
    long long x = 1000000, y = 0; // x的大小会影响精度,不能太大也不能太小,貌似10^6最好
    long long t = 0, r = a/pi()*180*512;
    while(r > 184320) r -= 184320;
    
    for(int i=0; i<16; i++){
        long long rx = x, ry = y;
        while(t < r){
            rx = x;
            ry = y;
            x = rx - (ry>>i);
            y = ry + (rx>>i);
            t += angle[i];
        }
        if(t == r){
            return (angle_t)y / sqrt(x*x + y*y);
        }else{
            t -= angle[i];
            x = rx;
            y = ry;
        }
    }
    return (angle_t)y / sqrt(x*x + y*y);
}

constexpr angle_t cos(angle_t a) {
    return sin(pi()/2 - a);
}

constexpr angle_t tan(angle_t a) {
    return sin(a) / cos(a);
}

constexpr angle_t cot(angle_t a) {
    return cos(a) / sin(a);
}

constexpr angle_t sec(angle_t a) {
    return 1 / cos(a);
}

constexpr angle_t csc(angle_t a) {
    return 1 / sin(a);
}

constexpr angle_t atan2(angle_t y, angle_t x) {
    constexpr int
    angle[] = {11520, 6801, 3593, 1824, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1};
    
    int x_new{}, y_new{};
    int angleSum = 0;
    
    int lx = x * 1000000;
    int ly = y * 1000000;
    
    for(int i = 0; i < 15; i++)
    {
        if(ly > 0)
        {
            x_new = lx + (ly >> i);
            y_new = ly - (lx >> i);
            lx = x_new;
            ly = y_new;
            angleSum += angle[i];
        }
        else
        {
            x_new = lx - (ly >> i);
            y_new = ly + (lx >> i);
            lx = x_new;
            ly = y_new;
            angleSum -= angle[i];
        }
    }
    return radians<angle_t>((angle_t)angleSum / (angle_t)256);
}

constexpr angle_t atan(angle_t a) {
    return atan2(a, 1);
}

constexpr angle_t acot2(angle_t x, angle_t y) {
    return atan2(y, x);
}

constexpr angle_t acot(angle_t a) {
    return atan2(1, a);
}

constexpr angle_t asin2(angle_t y, angle_t m) {
    angle_t x = sqrt(m*m - y*y);
    return atan2(y, x);
}

constexpr angle_t asin(angle_t a) {
    return asin2(a, 1);
}

constexpr angle_t acos2(angle_t x, angle_t m) {
    angle_t y = sqrt(m*m - x*x);
    return atan2(y, x);
}

constexpr angle_t acos(angle_t a) {
    return acos2(a, 1);
}

constexpr angle_t asec2(angle_t m, angle_t x) {
    return acos2(x, m);
}

constexpr angle_t asec(angle_t a) {
    return asec2(a, 1);
}

constexpr angle_t acsc2(angle_t m, angle_t y) {
    return asin2(y, m);
}

constexpr angle_t acsc(angle_t a) {
    return acsc2(a, 1);
}

// structures

#define VEC_MEM_FUNC_IMPL(N) \
template <unsigned int M> \
vec(const vec<T, M>& o) : vec{0} { \
    for (int i = 0; i < min(N, M); i++) asArray[i] = o[i]; \
} \
auto& operator[](unsigned int n) {return this->asArray[n];} /* non-const */ \
auto operator[](unsigned int n) const {return this->asArray[n];} \
auto value_ptr() {return asArray;} /* non-const */ \
auto value_ptr() const {return asArray;} \
auto operator+() const {return *this;} \
auto operator-() const {return vec<T, N>() - *this;} \
auto& operator+=(T k) { \
    for (int i=0; i<N; i++) asArray[i] += k;\
    return *this; \
} \
auto operator+(T k) const { \
    auto r = *this; \
    return r += k; \
} \
auto& operator-=(T k) { \
    for (int i=0; i<N; i++) asArray[i] -= k;\
    return *this; \
} \
auto operator-(T k) const { \
    auto r = *this; \
    return r -= k; \
} \
auto& operator*=(T k) { \
    for (int i=0; i<N; i++) asArray[i] *= k;\
    return *this; \
} \
auto operator*(T k) const { \
    auto r = *this; \
    return r *= k; \
} \
auto& operator/=(T k) { \
    for (int i=0; i<N; i++) asArray[i] /= k;\
    return *this; \
} \
auto operator/(T k) const { \
    auto r = *this; \
    return r /= k; \
} \
bool operator!=(vec<T, N> k) const { \
    for (int i=0; i<N; i++) \
        if (asArray[i] != k.asArray[i]) \
            return true; \
    return false; \
} \
bool operator==(vec<T, N> k) const {return !(*this != k);} \
constexpr operator mat<T, 1, N>() const; \
T length_squared() const { \
    T r{0}; \
    for (int i=0; i<N; i++) r += asArray[i]*asArray[i]; \
    return r; \
} \
T length() const {return sqrt(length_squared());} \
auto& normalize() {return *this = normalized();} \
auto normalized() const { \
    auto len = length(); \
    return *this / (len ? len : 1); \
}

template <class T, unsigned int W, unsigned int H>
struct mat;

template <class T, unsigned int N>
struct vec {
    vec() = default;
    
    T asArray[N]; // data
    
    VEC_MEM_FUNC_IMPL(N)
};

template <class T>
struct vec<T, 1> {
    constexpr vec(T x = T{0}) : x{x} {}
    
    union {
        struct { T x; };
        struct { T r; };
        struct { T i; };
        T asArray[1];
    };
    
    VEC_MEM_FUNC_IMPL(1)
};

template <class T>
struct vec<T, 2> {
    constexpr vec(T a = T{0}) : x{a}, y{a} {}
    constexpr vec(T x, T y) : x{x}, y{y} {}
    
    union {
        struct { T x, y; };
        struct { T r, g; };
        struct { T i, j; };
        T asArray[2];
    };
    
    VEC_MEM_FUNC_IMPL(2)
};

template <class T>
struct vec<T, 3> {
    constexpr vec(T a = T{0}) : x{a}, y{a}, z{a} {}
    constexpr vec(T x, T y, T z) : x{x}, y{y}, z{z} {}
    
    union {
        struct { T x, y, z; };
        struct { T r, g, b; };
        struct { T i, j, k; };
        T asArray[3];
    };
    
    VEC_MEM_FUNC_IMPL(3)
};

template <class T>
struct vec<T, 4> {
    constexpr vec(T a = T{0}) : x{a}, y{a}, z{a}, w{a} {}
    constexpr vec(T x, T y, T z, T w) : x{x}, y{y}, z{z}, w{w} {}
    
    union {
        struct { T x, y, z, w; };
        struct { T r, g, b, a; };
        struct { T i, j, k, l; };
        T asArray[4];
    };
    
    VEC_MEM_FUNC_IMPL(4)
};

template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N>& operator+=(vec<T, N>& v, const vec<T, Nk>& vk) {
    for (int i = 0; i < min(N, Nk); i++) v[i] += vk[i];
    return v;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N> operator+(const vec<T, N>& v, const vec<T, Nk>& vk) {
    auto r = v;
    return r += vk;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N>& operator-=(vec<T, N>& v, const vec<T, Nk>& vk) {
    for (int i = 0; i < min(N, Nk); i++) v[i] -= vk[i];
    return v;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N> operator-(const vec<T, N>& v, const vec<T, Nk>& vk) {
    auto r = v;
    return r -= vk;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N>& operator*=(vec<T, N>& v, const vec<T, Nk>& vk) {
    for (int i = 0; i < min(N, Nk); i++) v[i] *= vk[i];
    return v;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N> operator*(const vec<T, N>& v, const vec<T, Nk>& vk) {
    auto r = v;
    return r *= vk;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N>& operator/=(vec<T, N>& v, const vec<T, Nk>& vk) {
    for (int i = 0; i < min(N, Nk); i++) v[i] /= vk[i];
    return v;
}
template <class T, unsigned int N, unsigned int Nk>
constexpr vec<T, N> operator/(const vec<T, N>& v, const vec<T, Nk>& vk) {
    auto r = v;
    return r /= vk;
}

template <class T, unsigned int N>
vec<T, N> operator+(T k, vec<T, N> v) {
    return v + k;
}
template <class T, unsigned int N>
vec<T, N> operator*(T k, vec<T, N> v) {
    return v * k;
}

#undef VEC_MEM_FUNC_IMPL // prevent duplicate code

// normal vec type
using vec1 = vec<float, 1>;
using vec2 = vec<float, 2>;
using vec3 = vec<float, 3>;
using vec4 = vec<float, 4>;

using ivec1 = vec<int, 1>;
using ivec2 = vec<int, 2>;
using ivec3 = vec<int, 3>;
using ivec4 = vec<int, 4>;

using uivec1 = vec<unsigned int, 1>;
using uivec2 = vec<unsigned int, 2>;
using uivec3 = vec<unsigned int, 3>;
using uivec4 = vec<unsigned int, 4>;

using dvec1 = vec<double, 1>;
using dvec2 = vec<double, 2>;
using dvec3 = vec<double, 3>;
using dvec4 = vec<double, 4>;

template <class T, unsigned int W, unsigned int H>
struct mat {
    constexpr mat(T a = T{1}) {
        for (int i = 0; i < min(W, H); i++)
            element[i][i] = a;
    }
    constexpr mat(vec<T, H> e[W]) {
        for (int i = 0; i < W; i++) element[i] = e[i];
    }
    
    template <class...Args,
              class = utils::enable_if_t<(utils::is_same_v<utils::remove_cvref_t<Args>,
                                          vec<T, H>> && ...)>>
    constexpr mat(Args...args) {
        static_assert(sizeof...(Args) && sizeof...(Args) <= W, "illegal number of parameters");
        const vec<T, H> v[]{args...};
        for (int i = 0; i < sizeof...(Args); i++) element[i] = v[i];
    } // imitation aggregate initialization
    
    template <unsigned int W1, unsigned int H1>
    constexpr mat(const mat<T, W1, H1>& o) {
        for (int i = 0; i < min(W, W1); i++)
            element[i] = o[i];
    }
    
    vec<T, H> element[W]; // data
    
    auto value_ptr() {return element->value_ptr();} // non-const
    auto value_ptr() const {return element->value_ptr();}
    
    auto& operator[](unsigned int w) {return element[w];} // non-const
    auto operator[](unsigned int w) const {return element[w];}
    
    mat<T, W, H> transposed() const {
        mat<T, W, H> r;
        for(int i=0; i<H; i++)
            for(int j=0; j<W; j++)
                r[j][i] = element[i][j];
        return r;
    }
    
    mat<T, W-1, H-1> cofactor(int x, int y) const {
        mat<T, W-1, H-1> r(0.f);
        for(int i=0, rx=0; i<4; i++, rx++){
            if(i == x) continue;
            for(int j=0, ry=0; j<4; j++){
                if(j == y) continue;
                r[rx][ry++] = element[i][j];
            }
        }
        return r;
    } // 余子式
    
};

// normal mat type
using mat2 = mat<float, 2, 2>;
using mat3 = mat<float, 3, 3>;
using mat4 = mat<float, 4, 4>;

using dmat2 = mat<double, 2, 2>;
using dmat3 = mat<double, 3, 3>;
using dmat4 = mat<double, 4, 4>;

template <class T, unsigned int N>
constexpr vec<T, N>::operator mat<T, 1, N>() const {
    mat<T, 1, N> r{*this};
    return r;
}

template<class T, unsigned int W, unsigned int H, unsigned int M>
constexpr mat<T, W, H> operator*(const mat<T, M, H>& m1, const mat<T, W, M>& m2) {
    mat<T, W, H> r{T{0}};
    for (int i=0; i<H; i++)
        for (int j=0; j<W; j++)
            for (int k=0; k<M; k++)
                r[j][i] += m1[k][i] * m2[j][k];
    return r;
}

template<class T, unsigned int H, unsigned int N>
constexpr vec<T, H> operator*(const mat<T, N, H>& m, const vec<T, N>& v) {
    vec<T, H> r{};
    for (int i = 0; i < H; i++)
        for (int j = 0; j < N; j++)
            r[i] += m[j][i] * v[j];
    return r;
}

// 欧拉角
enum EARS{
    //Tait-Bryan Angle
    xyz, xzy, yxz, yzx, zxy, zyx,
    //Proper Euler Angle
    xyx, yxy, xzx, zxz, yzy, zyz
}; // 欧拉角旋转序列(Euler Angle Rotational Sequence)

using EulerAngle = vec<angle_t, 3>; // Euler Angle type

template<class T>
struct qua{
    qua() : w{T(1)} {}
    qua(T a) : w(a), x(a), y(a), z(a) {}
    qua(T w, T x, T y, T z) : w(w), x(x), y(y), z(z) {}
    qua(T s, vec<T, 3> v) : w(s), x(v.x), y(v.y), z(v.z) {}
    qua(vec<T, 3> u, angle_t angle) : qua<T>(T{cos(angle / 2)}, T{sin(angle / 2)} * u) {}
    qua(EulerAngle angles, EARS sequence);
    
    union {
        struct { T w, x, y, z; };
        struct { T l, i, j, k; };
    };
    
    T length_squared() const {return w*w + x*x + y*y + z*z;}
    T length() const {return sqrt(length_squared());}
    qua<T> conjugate() const {return {w, -vec<T, 3>{x, y, z}};}
    qua<T> inverse() const {return conjugate() / (length_squared());}
    
    qua<T> operator+() const {return *this;}
    qua<T> operator-() const {return qua<T>(T(0)) - *this;}
    qua<T> operator+(T k) const {return qua<T>(x + k, y + k, z + k, w + k);};
    qua<T>& operator+=(T k){x += k;y += k;z += k;w += k;return *this;}
    qua<T> operator-(T k) const {return qua<T>(x - k, y - k, z - k, w - k);};
    qua<T>& operator-=(T k) {x -= k;y -= k;z -= k;w -= k;return *this;}
    qua<T> operator*(T k) const {return qua<T>(x * k, y * k, z * k, w * k);};
    qua<T>& operator*=(T k) {x *= k;y *= k;z *= k;w *= k;return *this;}
    qua<T> operator/(T k) const {return qua<T>(x / k, y / k, z / k, w / k);};
    qua<T>& operator/=(T k) {x /= k;y /= k;z /= k;w /= k;return *this;}
    qua<T> operator+(qua<T> k) const {return qua<T>(x+k.x, y+k.y, z+k.z, w+k.w);}
    qua<T>& operator+=(qua<T> k) {x += k.x;y += k.y;z += k.z;w += k.w;return *this;}
    qua<T> operator-(qua<T> k) const {return qua<T>(x-k.x, y-k.y, z-k.z, w-k.w);}
    qua<T>& operator-=(qua<T> k) {x -= k.x;y -= k.y;z -= k.z;w -= k.w;return *this;}
    qua<T> operator/(qua<T> k) const {return qua<T>(x/k.x, y/k.y, z/k.z, w/k.w);}
    qua<T>& operator/=(qua<T> k) {x /= k.x;y /= k.y;z /= k.z;w /= k.w;return *this;}
    bool operator==(qua<T> k) const {return x == k.x && y == k.y && z == k.z && w == k.w;}
    bool operator!=(qua<T> k) const {return x != k.x || y != k.y || z != k.z || w != k.w;}
    qua<T> operator*(qua<T> k) const {
        T a = k.w, b = k.x, c = k.y, d = k.z;
        return {
            w*a - x*b - y*c - z*d,
            w*b + x*a + y*d - z*c,
            w*c - x*d + y*a + z*b,
            w*d + x*c - y*b + z*a
        };
    }
    qua<T>& operator*=(qua<T> k) {
        T a = k.w, b = k.x, c = k.y, d = k.z;
        w = w*a - x*b - y*c - z*d;
        x = w*b + x*a + y*d - z*c;
        y = w*c - x*d + y*a + z*b;
        z = w*d + x*c - y*b + z*a;
        return *this;
    }
};

template <class T>
qua<T>::qua(EulerAngle angles, EARS sequence) {
    angle_t p = angles[0], y = angles[1], r = angles[2];
    auto& rs = *this;
    
#define PMAT qua<T>(vec<T, 3>{1, 0, 0}, p)
#define YMAT qua<T>(vec<T, 3>{0, 1, 0}, y)
#define RMAT qua<T>(vec<T, 3>{0, 0, 1}, r)
    switch (sequence) {
        case xyz:
            rs = RMAT * YMAT * PMAT;
            break;
        case xzy:
            rs = YMAT * RMAT * PMAT;
            break;
        case yxz:
            rs = RMAT * PMAT * YMAT;
            break;
        case yzx:
            rs = PMAT * RMAT * YMAT;
            break;
        case zxy:
            rs = YMAT * PMAT * RMAT;
            break;
        case zyx:
            rs = PMAT * YMAT * RMAT;
            break;
        case xyx:
            rs = PMAT * YMAT * PMAT;
            break;
        case yxy:
            rs = YMAT * PMAT * YMAT;
            break;
        case xzx:
            rs = PMAT * RMAT * PMAT;
            break;
        case zxz:
            rs = RMAT * PMAT * RMAT;
            break;
        case yzy:
            rs = YMAT * RMAT * YMAT;
            break;
        case zyz:
            rs = RMAT * YMAT * RMAT;
            break;
    }
#undef PMAT
#undef YMAT
#undef RMAT
}

// normal quat type
using quat = qua<float>;

// useful funstions

template <class T, unsigned int N>
constexpr T dot(vec<T, N> v1, vec<T, N> v2) {
    T r{0};
    for (int i=0; i<N; i++) r += v1[i] * v2[i];
    return r;
}

template <class T>
constexpr vec<T, 3> cross(vec<T, 3> v1, vec<T, 3> v2){
    mat<T, 3, 3> r(0.f);
    r[2][1]-= r[1][2] = v1.x;
    r[2][0]-= r[0][2]-= v1.y;
    r[1][0]-= r[0][1] = v1.z;
    return r * v2;
}

template <class T, unsigned int N>
constexpr angle_t angle(vec<T, N> v1, vec<T, N> v2){
    return acos(dot(v1, v2) / v1.length() / v2.length());
}

template <class T, unsigned int N>
constexpr vec<T, N> reflect(vec<T, N> ori, vec<T, N> normal){
    return ori - 2 * mathpls::dot(ori, normal) * normal;
}

template <class T, unsigned int N>
constexpr vec<T, N> project(vec<T, N> len, vec<T, N> dir) {
    return dir * (dot(len, dir) / dir.length_squared());
}

template <class T, unsigned int N>
constexpr vec<T, N> perpendicular(vec<T, N> len, vec<T, N> dir) {
    return len - project(len, dir);
}

// transformation functions

template <class T, unsigned int N>
mat<T, N+1, N+1> translate(vec<T, N> v, mat<T, N+1, N+1> ori = {}) {
    for (int i = 0; i < N; i++) ori[N][i] += v[i];
    return ori;
}

template <class T = float> // this might be unable to derive
mat<T, 3, 3> rotate(angle_t angle, mat<T, 3, 3> ori = {}) {
    mat<T, 3, 3> r{};
    r[0][0] = r[1][1] = cos(angle);
    r[0][1]-= r[1][0]-= sin(angle);
    return r * ori;
}

template <class T>
mat<T, 4, 4> rotate(vec<T, 3> axis, angle_t angle, mat<T, 3, 3> ori = {}) {
    const T& x = axis.x, y = axis.y, z = axis.z;
    angle_t sa = sin(angle), ca = cos(angle);
    angle_t bca = 1 - ca;
    
    mat<T, 4, 4> r = {
        vec<T, 4>{ca + x*x*bca, sa*z + bca*x*y, -sa*y + bca*x*z, 0},
        vec<T, 4>{-sa*z + bca*x*y, ca + y*y*bca, sa*x + bca*y*z, 0},
        vec<T, 4>{sa*y + bca*x*z, -sa*x + bca*y*z, ca + z*z*bca, 0},
        vec<T, 4>{0, 0, 0, 1}
    };
    
    return r * ori;
}

template <class T>
mat<T, 4, 4> rotate(EulerAngle angles, EARS sequence, mat<T, 4, 4> ori = {}){
    angle_t p = angles[0], y = angles[1], r = angles[2];
    mat4 rs(1);
    
#define PMAT rotate(vec<T, 3>{1, 0, 0}, p)
#define YMAT rotate(vec<T, 3>{0, 1, 0}, y)
#define RMAT rotate(vec<T, 3>{0, 0, 1}, r)
    switch (sequence) {
        case xyz:
            rs = RMAT * YMAT * PMAT;
            break;
        case xzy:
            rs = YMAT * RMAT * PMAT;
            break;
        case yxz:
            rs = RMAT * PMAT * YMAT;
            break;
        case yzx:
            rs = PMAT * RMAT * YMAT;
            break;
        case zxy:
            rs = YMAT * PMAT * RMAT;
            break;
        case zyx:
            rs = PMAT * YMAT * RMAT;
            break;
        case xyx:
            rs = PMAT * YMAT * PMAT;
            break;
        case yxy:
            rs = YMAT * PMAT * YMAT;
            break;
        case xzx:
            rs = PMAT * RMAT * PMAT;
            break;
        case zxz:
            rs = RMAT * PMAT * RMAT;
            break;
        case yzy:
            rs = YMAT * RMAT * YMAT;
            break;
        case zyz:
            rs = RMAT * YMAT * RMAT;
            break;
    }
#undef PMAT
#undef YMAT
#undef RMAT
    
    return rs * ori;
}

template <class T, unsigned int N>
mat<T, N, N> scale(vec<T, N-1> s, mat<T, N, N> ori = {}) {
    mat<T, N, N> r{};
    for (int i = 0; i < N-1; i++)
        r[i][i] = s[i][i];
    return r * ori;
}

template<class T>
mat<T, 4, 4> rotate(qua<T> q){
    const T a = q.w, b = q.x, c = q.y, d = q.z;
    mat<T, 4, 4> m = {
        vec<T, 4>{1 - 2*c*c - 2*d*d, 2*b*c + 2*a*d, 2*b*d - 2*a*c, 0},
        vec<T, 4>{2*b*c - 2*a*d, 1 - 2*b*b - 2*d*d, 2*a*b + 2*c*d, 0},
        vec<T, 4>{2*a*c + 2*b*d, 2*c*d - 2*a*b, 1 - 2*b*b - 2*c*c, 0},
        vec<T, 4>{0, 0, 0, 1}
    };
    return m;
}

template <class T>
mat<T, 4, 4> lookAt(vec<T, 3> eye, vec<T, 3> target, vec<T, 3> up){
    vec<T, 4> d = (eye - target).normalized();
    vec<T, 4> r = cross(up, d).normalized();
    vec<T, 4> u = cross(d, r).normalized();
    mat<T, 4, 4> m = {
        r, u, d,
        vec<T, 4>{0, 0, 0, 1}
    };
    return m.transposed() * translate(-eye);
}

template <class T>
mat<T, 4, 4> ortho(T l, T r, T b, T t){
    float m = {
        vec<T, 4>{2/(r - l), 0, 0, 0},
        vec<T, 4>{0, 2/(t - b), 0, 0},
        vec<T, 4>{0, 0,        -1, 0},
        vec<T, 4>{(l+r)/(l-r), (b+t)/(b-t), 0, 1}
    };
    return m;
}

template <class T>
mat<T, 4, 4> ortho(T l, T r, T b, T t, T n, T f){
    mat<T, 4, 4> m = {
        vec<T, 4>{2/(r - l), 0, 0, 0},
        vec<T, 4>{0, 2/(t - b), 0, 0},
        vec<T, 4>{0, 0, 2/(n - f), 0},
        vec<T, 4>{(l+r)/(l-r), (b+t)/(b-t), (f+n)/(n-f), 1}
    };
    return m;
}

template <class T>
mat<T, 4, 4> perspective(T fov, T asp, T near, T far){
    mat<T, 4, 4> m = {
        vec<T, 4>{cot(fov/2)/asp, 0, 0, 0},
        vec<T, 4>{0, cot(fov/2),     0, 0},
        vec<T, 4>{0, 0, (far + near)/(near - far),-1},
        vec<T, 4>{0, 0, (2*far*near)/(near - far), 0}
    };
    return m;
}

namespace random {

struct rand_sequence {
private:
    unsigned int m_index;
    unsigned int m_intermediateOffset;

    static unsigned int permuteQPR(unsigned int x) {
        static const unsigned int prime = 4294967291u;
        if (x >= prime)
            return x;  // The 5 integers out of range are mapped to themselves.
        unsigned int residue = ((unsigned long long) x * x) % prime;
        return (x <= prime / 2) ? residue : prime - residue;
    }

public:
    rand_sequence(unsigned int seedBase, unsigned int seedOffset) {
        m_index = permuteQPR(permuteQPR(seedBase) + 0x682f0161);
        m_intermediateOffset = permuteQPR(permuteQPR(seedOffset) + 0x46790905);
    }
    rand_sequence(unsigned int seed) : rand_sequence(seed, seed + 1) {}

    unsigned int next() {
        return permuteQPR((permuteQPR(m_index++) + m_intermediateOffset) ^ 0x5bf03635);
    }
    
    unsigned int operator()() {
        return next();
    }
};

struct mt19937 {
    mt19937(unsigned int seed) {
        mt[0] = seed;
        for(int i=1;i<624;i++)
            mt[i] = static_cast<unsigned int>(1812433253 * (mt[i - 1] ^ mt[i - 1] >> 30) + i);
    }
    
    unsigned int operator()() {
        return extract_number();
    }
    
private:
    unsigned int mt[624];
    unsigned int mti{0};
    
    unsigned int extract_number() {
        if(mti == 0) twist();
        unsigned long long y = mt[mti];
        y = y ^ y >> 11;
        y = y ^ (y << 7 & 0x9D2C5680);
        y = y ^ (y << 15 & 0xEFC60000);
        y = y ^ y >> 18;
        mti = (mti + 1) % 624;
        return static_cast<unsigned int>(y);
    }
    
    void twist() {
        for(int i=0;i<624;i++) {
            // 高位和低位级联
            auto y = static_cast<unsigned int>((mt[i] & 0x80000000) + (mt[(i + 1) % 624] & 0x7fffffff));
            mt[i] = (y >> 1) ^ mt[(i + 397) % 624];
            if(y % 2 != 0) mt[i] = mt[i] ^ 0x9908b0df; // 如果最低为不为零
        }
    }
};

template<class T>
struct uniform_real_distribution {
    uniform_real_distribution(T a, T b) : a(a), b(b) {}
    
    template<class E>
    T operator()(E e) const {
        return a + (b - a) * e() / 0xffffffff;
    }
    
private:
    T a, b;
};
 
inline unsigned int rand() {
    static mt19937 e{114514 ^ 1919810};
    return e();
}

/**
 * \result a random number in the range from 0 to 1
 */
template <class T = double>
T rand01() {
    return static_cast<T>(rand()) / 0xffffffff;
}

/**
 * \result a random number in the range from -1 to 1
 */
template <class T = double>
T rand11() {
    return rand01<T>() * 2 - 1;
}

template <class T, unsigned int N>
struct rand_vec_fn {
    constexpr rand_vec_fn() = default;
    
    /**
     * \result a normalized random vector
     */
    vec<T, N> operator()() const {
        vec<T, N> r;
        for (auto& i : r.asArray) i = rand11<T>();
        return r.normalized();
    }
};

template <class T, unsigned int N>
constexpr auto rand_vec = rand_vec_fn<T, N>{};

constexpr auto rand_vec2 = rand_vec<float, 2>;
constexpr auto rand_vec3 = rand_vec<float, 3>;

constexpr auto rand_dvec2 = rand_vec<double, 2>;
constexpr auto rand_dvec3 = rand_vec<double, 3>;

}

} // mathpls
