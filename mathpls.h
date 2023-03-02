#ifndef _MATHPLS_H_
#define _MATHPLS_H_
namespace mathpls{

template<class T>
constexpr T max(T a, T b){
    return a>b ? a:b;
}

template<class T>
constexpr T min(T a, T b){
    return a<b ? a:b;
}

template<class T> // 返回第二大的,你也可以当成,是否在范围内,否则返回最大或最小值
constexpr T mid(T min, T a, T max){
    return (min<(a<max?a:max)?(a<max?a:max):min<(a>max?a:max)?min:(a>max?a:max));
}

template<class T>
constexpr T abs(T a){
    return a > 0 ? a : -a;
}

constexpr static long double PI = 3.14159265358979323846264338327950288;

template<class T = float>
constexpr T radians(long double angle){
    return angle / (long double)180 * PI;
}

template<class T = float>
constexpr T pow(T x, int n){
    T r = x;
    if(n>0){
        while(--n) r *= x;
    } else {
        do{r /= x;}while(n++);
    }
    return r;
}

long double sqrt(long double x){
    if (x == 1 || x == 0)
        return x;
    double temp = x / 2;
    while (abs(temp - (temp + x / temp) / 2) > 0.000001)
        temp = (temp + x / temp) / 2;
    return temp;
}

long double sin(long double a){
    if(a < 0){
        return -sin(-a); // sin(-a) = -sin(a)
    }
    while(a > PI*2) a -= PI*2;
    long double angle[] = {PI/4,PI/8,PI/16,
        PI/32,PI/64,PI/128,
        PI/256,PI/512,PI/1024,
        PI/2048,PI/4096,PI/8192,PI/16384};
    long double tang[]={1,0.4142135623731,0.19891236737966,
        0.098491403357164,0.049126849769467,
        0.024548622108925,0.012272462379566,
        0.0061360001576234,0.0030679712014227,
        0.0015339819910887,0.00076699054434309,
        0.00038349521577144,0.00019174760083571};
    
    if(a <= (PI/16384)){
        return a; //因为a的值太小，sin a 约等于 a
    }else{
        //开始 CORDIC 算法
        long double x = 10;
        long double y = 0;
        long double theta = 0;
        for(int i = 0;i < 13;i ++){  //开始迭代
            long double orix = x, oriy = y;
            while(theta < a){ //当前角度小于a
                orix = x;
                oriy = y;
                //坐标旋转
                x = orix - tang[i] * oriy;
                y = tang[i] * orix + oriy;
                theta += angle[i];
            }
            if(theta == a){
                return (y/sqrt((x*x+y*y)));
            }else{
                //旋转的弧度超过了a，退回原来增加的角度，同时进入下一次迭代
                theta -= angle[i];
                x = orix;
                y = oriy;
            }
        }
        return (y/sqrt((x*x+y*y)));
    }
}

long double cos(long double a){
    return sin(PI/2 - a);
}

long double tan(long double a){
    return sin(a) / cos(a);
}

long double cot(long double a){
    return cos(a) / sin(a);
}

long double sec(long double a){
    return 1 / cos(a);
}

long double csc(long double a){
    return 1 / sin(a);
}

long double atan2(long double y, long double x)
{
    int angle[] = {11520, 6801, 3593, 1824, 916, 458, 229, 115, 57, 29, 14, 7, 4, 2, 1};
    
    int x_new, y_new;
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
    return radians((long double)angleSum / (long double)256);
}

long double atan(long double a){
    return atan2(a, 1);
}

long double acot2(long double x, long double y){
    return atan2(y, x);
}

long double acot(long double a){
    return atan2(1, a);
}

long double asin2(long double y, long double m){
    long double x = sqrt(m*m - y*y);
    return atan2(y, x);
}

long double asin(long double a){
    return asin2(a, 1);
}

long double acos2(long double x, long double m){
    long double y = sqrt(m*m - x*x);
    return atan2(y, x);
}

long double acos(long double a){
    return acos2(a, 1);
}

long double asec2(long double m, long double x){
    return acos2(x, m);
}

long double asec(long double a){
    return asec2(a, 1);
}

long double acsc2(long double m, long double y){
    return asin2(y, m);
}

long double acsc(long double a){
    return acsc2(a, 1);
}

struct vec1{
    vec1() : x(1) {}
    vec1(float x) : x(x) {}
    float x;
    
    vec1 operator*(float k){return vec1(x * k);}
    vec1 operator*=(float k){x *= k;return *this;}
    vec1 operator/(float k){return vec1(x / k);}
    vec1 operator/=(float k){x /= k;return *this;}
    vec1 operator+(vec1 k){return vec1(x+k.x);}
    vec1 operator+=(vec1 k){x += k.x;return *this;}
    vec1 operator-(vec1 k){return vec1(x-k.x);}
    vec1 operator-=(vec1 k){x -= k.x;return *this;}
    
    float length() {return abs(x);}
    vec1 normalize() {return *this / length();}
};// 真的有人用vec1吗?
struct vec2{
    vec2() : x(1), y(0) {}
    vec2(float x, float y) : x(x), y(y) {}
    vec2(vec1 v1, float y) : x(v1.x), y(y) {}
    float x, y;
    
    vec2 operator*(float k){return vec2(x * k, y * k);}
    vec2 operator*=(float k){x *= k;y *= k;return *this;}
    vec2 operator/(float k){return vec2(x / k, y / k);}
    vec2 operator/=(float k){x /= k;y /= k;return *this;}
    vec2 operator+(vec2 k){return vec2(x+k.x, y+k.y);}
    vec2 operator+=(vec2 k){x += k.x;y += k.y;return *this;}
    vec2 operator-(vec2 k){return vec2(x-k.x, y-k.y);}
    vec2 operator-=(vec2 k){x -= k.x;y -= k.y;return *this;}
    
    float length() {return sqrt(x*x + y*y);}
    vec2 normalize() {return *this / length();}
};
struct vec3{
    vec3() : x(1), y(0), z(0) {}
    vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    vec3(vec2 v2, float z) : x(v2.x), y(v2.y), z(z) {}
    float x, y, z;
    
    vec3 operator*(float k){return vec3(x * k, y * k, z * k);}
    vec3 operator*=(float k){x *= k;y *= k;z *= k;return *this;}
    vec3 operator/(float k){return vec3(x / k, y / k, z / k);}
    vec3 operator/=(float k){x /= k;y /= k;z /= k;return *this;}
    vec3 operator+(vec3 k){return vec3(x+k.x, y+k.y, z+k.z);}
    vec3 operator+=(vec3 k){x += k.x;y += k.y;z += k.z;return *this;}
    vec3 operator-(vec3 k){return vec3(x-k.x, y-k.y, z-k.z);}
    vec3 operator-=(vec3 k){x -= k.x;y -= k.y;z -= k.z;return *this;}
    
    float length() {return sqrt(x*x + y*y + z*z);}
    vec3 normalize() {return *this / length();}
};
struct vec4{
    vec4() : w(0), x(1), y(0), z(0) {}
    vec4(float w, float x, float y, float z) : w(w), x(x), y(y), z(z) {}
    vec4(float w, vec3 v3) : w(w), x(v3.x), y(v3.y), z(v3.z) {}
    float w, x, y, z;
    
    vec4 operator*(float k){return vec4(w * k, x * k, y * k, z * k);};
    vec4 operator*=(float k){w *= k;x *= k;y *= k;z *= k;return *this;}
    vec4 operator/(float k){return vec4(w / k, x / k, y / k, z / k);};
    vec4 operator/=(float k){w /= k;x /= k;y /= k;z /= k;return *this;}
    vec4 operator+(vec4 k){return vec4(w+k.w, x+k.x, y+k.y, z+k.z);}
    vec4 operator+=(vec4 k){w += k.w;x += k.x;y += k.y;z += k.z;return *this;}
    vec4 operator-(vec4 k){return vec4(w-k.w, x-k.x, y-k.y, z-k.z);}
    vec4 operator-=(vec4 k){w -= k.w;x -= k.x;y -= k.y;z -= k.z;return *this;}
    
    float length() {return sqrt(w*w + x*x + y*y + z*z);}
    vec4 normalize() {return *this / length();}
};

template<int H, int W>
struct mat{
    mat(float m = 1.f){
        for(int i=0; i<min(h,w); i++) element[i][i] = m;
    }
    mat(float e[H][W]){
        memcpy(element, e, sizeof(float) * H * W);
    }
    float element[H][W] = {0};
    int h = H, w = W;
    
    float* operator[](unsigned int x){return element[x];}
    
    mat<H, W> operator*(mat<H, W> m){
        mat<H, W> result(0.f);
        for(int i=0;i<H;i++){
            for(int j=0;j<W;j++){
                for(int k=0;k<min(H, W);k++){
                    result[i][j] += element[i][k] * m[k][j];
                }
            }
        }
        return result;
    }
    mat<H, W> operator*(float k){
        mat<H, W> r = *this;
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                r[i][j] *= k;
            }
        }
        return r;
    }
    mat<H, W> operator*=(float k){
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                element[i][j] *= k;
            }
        }
    }
    mat<H, W> operator/(float k){
        mat<H, W> r = *this;
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                r[i][j] /= k;
            }
        }
        return r;
    }
    mat<H, W> operator/=(float k){
        for(int i = 0; i < H; i++){
            for(int j = 0; j < W; j++){
                element[i][j] /= k;
            }
        }
    }
    
    // 余子式
    mat<H-1, W-1> cofactor(int x, int y){
        int rx=0, ry=0;
        mat<H-1, W-1> r(0.f);
        for(int i=0; i<4; i++){
            if(i == x) continue;
            for(int j=0; j<4; j++){
                if(j == y) continue;
                r[rx][ry++] = element[i][j];
            }
            rx++;
            ry = 0;
        }
        return r;
    }
};

using mat2 = mat<2, 2>;
using mat3 = mat<3, 3>;
using mat4 = mat<4, 4>;

vec2 operator*(mat2 m, vec2 v){
    return vec2(m[0][0]*v.x+m[0][1]*v.y, m[1][0]*v.x+m[1][1]*v.y);
}
vec3 operator*(mat3 m, vec3 v){
    return vec3(m[0][0]*v.x+m[0][1]*v.y+m[0][2]*v.z,
                m[1][0]*v.x+m[1][1]*v.y+m[1][2]*v.z,
                m[2][0]*v.x+m[2][1]*v.y+m[2][2]*v.z);
}
vec4 operator*(mat4 m, vec4 v){
    return vec4(m[0][0]*v.w+m[0][1]*v.x+m[0][2]*v.y+m[0][3]*v.z,
                m[1][0]*v.w+m[1][1]*v.x+m[1][2]*v.y+m[1][3]*v.z,
                m[2][0]*v.w+m[2][1]*v.x+m[2][2]*v.y+m[2][3]*v.z,
                m[3][0]*v.w+m[3][1]*v.x+m[3][2]*v.y+m[3][3]*v.z);
}

template<int H, int W>
mat<H, W> operator*(float k, mat<H, W> m){
    mat<H, W> r = m;
    for(int i = 0; i < H; i++){
        for(int j = 0; j < W; j++){
            r[i][j] *= k;
        }
    }
    return r;
}

// 行列式值, 只有方阵能求, 所以这里只实现那三个方阵, 其他更多的你只能自己实现
float determinant(mat2 m){
    return m[0][0]*m[1][1] - m[0][1]*m[1][0];
}
float determinant(mat3 m){
    float r = 0;
    for(int i=0; i<3; i++){
        r += determinant(m.cofactor(2, i));
    }
    return r;
}
float determinant(mat4 m){
    float r = 0;
    for(int i=0; i<4; i++){
        r += determinant(m.cofactor(3, i));
    }
    return r;
}

float dot(vec2 v1, vec2 v2){
    return v1.x*v2.x + v1.y*v2.y;
}
float dot(vec3 v1, vec3 v2){
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}
float dot(vec4 v1, vec4 v2){
    return v1.w*v2.w + v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

// 如果你想问为什么只有vec3, 那你就先回去读读高中
vec3 cross(vec3 v1, vec3 v2){
    mat3 r(0.f);
    r[1][2] = r[2][1] = v1.x;
    r[0][2] -= r[2][0] -= v1.y;
    r[0][1] = r[1][0] -= v1.z;
    return r * v2;
}// 欢迎各位三体人来实现vec4的外积

// 以下开始矩阵变换, 我相信不会有人用一维的变换，所以没有mat2

mat3 translate(mat3 ori, vec2 t){
    mat3 r(1);
    r[0][2] = t.x;
    r[1][2] = t.y;
    return r * ori;
}
mat3 translate(vec2 t){
    mat3 r(1);
    r[0][2] = t.x;
    r[1][2] = t.y;
    return r;
}
mat4 translate(mat4 ori, vec3 t){
    mat4 r(1);
    r[1][0] = t.x;
    r[2][0] = t.y;
    r[3][0] = t.z;
    return r * ori;
}
mat4 translate(vec3 t){
    mat4 r(1);
    r[1][0] = t.x;
    r[2][0] = t.y;
    r[3][0] = t.z;
    return r;
}

mat3 rotate(mat3 ori, long double angle){
    mat3 r(1);
    r[0][0] = r[1][1] = cos(angle);
    r[1][0] -= r[0][1] -= sin(angle);
    return r * ori;
}
mat3 rotate(long double angle){
    mat3 r(1);
    r[0][0] = r[1][1] = cos(angle);
    r[1][0] -= r[0][1] -= sin(angle);
    return r;
}
mat4 rotate(mat4 ori, long double angle, vec3 u){
    // 暂未实现
    return mat4();
}

}
#endif
