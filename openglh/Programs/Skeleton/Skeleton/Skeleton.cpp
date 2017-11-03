//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kovács Zsolt
// Neptun : O1UFAY
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif


const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...


using namespace std;
// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec3 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, vertexPosition.z, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}
	vec4 operator*(const float& f) {
		vec4 result;
		for (int i = 0; i < 4; i++) {
			result.v[i] = v[i] * f;
		}
		return result;
	}
	vec4 operator*(const vec4& a) {
		vec4 result;
		for (int i = 0; i < 4; i++) {
			result.v[i] = v[i] * a.v[i];
		}
		return result;
	}
	vec4 operator+=(const vec4& a) {
		for (int i = 0; i < 4; i++) {
			v[i] += a.v[i];
		}
		return *this;
	}
	vec4 operator+(const vec4& a) {
		vec4 result(0, 0, 0, 0);
		for (int i = 0; i < 4; i++) {
			result.v[i] = v[i] + a.v[i];
		}
		return result;
	}
	vec4 operator-(const vec4& a) {
		vec4 result(0, 0, 0, 0);
		for (int i = 0; i < 4; i++) {
			result.v[i] = v[i] - a.v[i];
		}
		return result;
	}

	vec4 operator/(const vec4& a) {
		vec4 result(0, 0, 0, 0);
		for (int i = 0; i < 4; i++) {
			if (a.v[i] != 0)
				result.v[i] = v[i] / a.v[i];
		}
		return result;
	}

	vec4 operator/(const float& a) {
		vec4 result(0, 0, 0, 0);
		for (int i = 0; i < 4; i++) {
			result.v[i] = v[i] / a;
		}
		return result;
	}
	vec4 normal() {
		vec4 res = vec4(v[0], v[1], v[2], v[3]) / sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		return res;
	}


	void write() {
		printf("%f,\t%f,\t%f", v[0], v[1], v[2]);
	}

};

// 2D camera
struct Camera {
	float wCx, wCy, wCz;	// center in world coordinates
	float wWx, wWy, wWz;	// width and height in world coordinates
	float a, b, c;
public:
	Camera() {
		Animate(0);
		a = b = c = 0;
	}


	mat4 Rx() {
		return mat4(1, 0, 0, 0,
			0, cos(a), -sin(a), 0,
			0, sin(a), cos(a), 0,
			0, 0, 0, 1);
	}
	mat4 Ry() {
		return mat4(cos(b), 0, sin(b), 0,
			0, 1, 0, 0,
			-sin(b), 0, cos(b), 0,
			0, 0, 0, 1);
	}
	mat4 Rz() {
		return mat4(cos(c), -sin(c), 0, 0,
			sin(c), cos(c), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
	}

	mat4 Rxyz() {
		return Rx()*Ry()*Rz();
	}
	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			-wCx, -wCy, -wCz, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2 / wWx, 0, 0, 0,
			0, 2 / wWy, 0, 0,
			0, 0, 2 / wWz, 0,
			0, 0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			wCx, wCy, wCz, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx / 2, 0, 0, 0,
			0, wWy / 2, 0, 0,
			0, 0, wWz / 2, 0,
			0, 0, 0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wCz = 0;
		wWx = 20;
		wWy = 20;
		wWz = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

float R1 = 6.f;
float R2 = 3.f;


class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
public:
	Triangle() {
		Animate(0);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

									// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		static float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertexCoords), // number of the vbo in bytes
			vertexCoords,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
								   // Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0 
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed

						  // vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		static float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

																							// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
									   // Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
			0, sy, 0, 0,
			0, 0, 0, 0,
			0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 0, 0,
			wTx, wTy, 0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V()* camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[100]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		if (nVertices >= 20) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices] = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}
};

class Torus {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[400000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	Torus() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

		AddPoints();


	}
	vec4 uv_to_xyz(float u, float v) {
		vec4 result;
		float alfa = u / R2;
		float beta = v / R1;

		float x = (R1 + R2*cos(alfa))*cos(beta);
		float y = (R1 + R2*cos(alfa))*sin(beta);
		float z = R2*sin(alfa);
		result = vec4(x, y, z, 1);
		return result;
	}

	float distance(float Au, float Av, float Bu, float Bv) {
		float result = 0;

		float A_alfa = Au / R2;
		float A_beta = Av / R1;

		float B_alfa = Bu / R2;
		float B_beta = Bv / R1;

		float nSamples = 500.f;
		float density_alfa = (B_alfa - A_alfa) / nSamples;
		float density_beta = (B_beta - A_beta) / nSamples;

		float alfa, beta;
		alfa = A_alfa;
		beta = A_beta;

		for (float i = 0; i < nSamples; i++) {


			float x = (R1 + R2*cos(alfa))*cos(beta);
			float y = (R1 + R2*cos(alfa))*sin(beta);
			float z = R2*sin(alfa);

			vec4 p0 = vec4(x, y, z, 1);

			vertexData[6 * nVertices] = x;
			vertexData[6 * nVertices + 1] = y;
			vertexData[6 * nVertices + 2] = z;
			vertexData[6 * nVertices + 3] = 0; // red
			vertexData[6 * nVertices + 4] = 1; // green
			vertexData[6 * nVertices + 5] = 0; // blue

			nVertices++;


			alfa += density_alfa;
			beta += density_beta;

			x = (R1 + R2*cos(alfa))*cos(beta);
			y = (R1 + R2*cos(alfa))*sin(beta);
			z = R2*sin(alfa);

			vec4 p1 = vec4(x, y, z, 1);
			vec4 dv = p1 - p0;
			float d = sqrtf(dv.v[0] * dv.v[0] + dv.v[1] * dv.v[1] + dv.v[2] * dv.v[2]);
			result += d;

		}
		return result;

	}
	void AddPoints() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);


		float x;
		float y;
		float z;

		vec4 A = vec4(0,0,0);
		vec4 B = vec4(R2 * 2 * M_PI / 2, R1 * 2 * M_PI / 2);
		vec4 C = vec4();
		for (float i = 0; i < 2 * M_PI; i += 0.01f) {
			for (float j = 0; j < 2 * M_PI; j += 0.1f) {

				x = (R1 + R2*cos(j))*cos(i);
				y = (R1 + R2*cos(j))*sin(i);
				z = R2*sin(j);

				vec4 wVertex = vec4(x, y, z, 1);
				// fill interleaved data
				vertexData[6 * nVertices] = wVertex.v[0];
				vertexData[6 * nVertices + 1] = wVertex.v[1];
				vertexData[6 * nVertices + 2] = wVertex.v[2];
				vertexData[6 * nVertices + 3] = 1; // red
				vertexData[6 * nVertices + 4] = 0; // green
				vertexData[6 * nVertices + 5] = 0; // blue

				nVertices++;

			}

		}

		//float res=distance(0, 0, 0, 2 * R1*M_PI);
		//printf("\ndistance=%f\n", res);

		for (int i = 0; i < 2000; i++) {
			float rand1 = (rand() % 2000) / 100.f - 10.f;
			float rand2 = (rand() % 2000) / 100.f - 10.f;

		}

		printf("%d", nVertices * 6);
		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 6 * sizeof(float), vertexData, GL_STATIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P() * camera.Rxyz();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_POINTS, 0, nVertices);
		}
	}
};
// The virtual world: collection of two objects

class Bezier {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[100000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
	vector<vec4> controlPoints;
	int nPoints;
	vec4 a, b, c;
	vec4 PAB, PBC, PCA;
public:
	Bezier() {
		nVertices = 0;
		nPoints = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));


	}
	float distance(float Au, float Av, float Bu, float Bv) {
		float result = 0;

		float A_alfa = Au / R2;
		float A_beta = Av / R1;

		float B_alfa = Bu / R2;
		float B_beta = Bv / R1;

		float nSamples = 500.f;
		float density_alfa = (B_alfa - A_alfa) / nSamples;
		float density_beta = (B_beta - A_beta) / nSamples;

		float alfa, beta;
		alfa = A_alfa;
		beta = A_beta;

		for (float i = 0; i < nSamples; i++) {

			float x = (R1 + R2*cos(alfa))*cos(beta);
			float y = (R1 + R2*cos(alfa))*sin(beta);
			float z = R2*sin(alfa);

			vec4 p0 = vec4(x, y, z, 1);

			alfa += density_alfa;
			beta += density_beta;

			x = (R1 + R2*cos(alfa))*cos(beta);
			y = (R1 + R2*cos(alfa))*sin(beta);
			z = R2*sin(alfa);

			vec4 p1 = vec4(x, y, z, 1);
			vec4 dv = p1 - p0;
			float d = sqrtf(dv.v[0] * dv.v[0] + dv.v[1] * dv.v[1] + dv.v[2] * dv.v[2]);
			result += d;

		}
		return result;

	}
	float B(int i, float t,vector<vec4> cps) {
		int n = cps.size() - 1; // n deg polynomial = n+1 pts!
		float choose = 1;
		for (int j = 1; j <= i; j++) choose *= (float)(n - j + 1) / j;
		return choose * pow(t, i) * pow(1 - t, n - i);
	}

	vec4 r(float t,vector<vec4> cps) {
		vec4 rr(0, 0, 0);
		for (int i = 0; i < cps.size(); i++) rr += cps[i] * B(i, t,cps);
		return rr;
	}

	void loadPoints(vector<vec4> cps) {
		vec4 r1;
		for (float i = 0; i <= 1.f; i += 0.01f) {
			r1 = r(i, cps);

			// fill interleaved data
			vertexData[6 * nVertices] = r1.v[0];
			vertexData[6 * nVertices + 1] = r1.v[1];
			vertexData[6 * nVertices + 2] = 0;
			vertexData[6 * nVertices + 3] = 1;
			vertexData[6 * nVertices + 4] = 1;
			vertexData[6 * nVertices + 5] = 0;
			nVertices++;

			r1 = click_to_xyz(r1);

			vertexData[6 * nVertices] = r1.v[0];
			vertexData[6 * nVertices + 1] = r1.v[1];
			vertexData[6 * nVertices + 2] = r1.v[2];
			vertexData[6 * nVertices + 3] = 0;
			vertexData[6 * nVertices + 4] = 1;
			vertexData[6 * nVertices + 5] = 0;
			nVertices++;
		}
	}
	float calculateDistance(vector<vec4> cps) {
		float result=0;
		vec4 r1;
		vec4 r_next;
		vec4 tempVec;
		for (float i = 0; i < 1.f; i += 0.001f) {
			r1 = r(i, cps);
			r_next = r(i + 0.0001f, cps);

			r1 = click_to_xyz(r1);
			r_next = click_to_xyz(r_next);

			tempVec=r_next - r1;
			result += sqrtf(tempVec.v[0] * tempVec.v[0] + tempVec.v[1] * tempVec.v[1] + tempVec.v[2] * tempVec.v[2]);
		}
		return result;

	}
	vec4 click_to_xyz(vec4 vec) {
		float u1 = (vec.v[0] + 10.f) / 20.f * 2 * M_PI*R2;
		float v1 = (vec.v[1] + 10.f) / 20.f * 2 * M_PI*R1;
		return uv_to_xyz(u1, v1);
	}
	vec4 uv_to_xyz(float u, float v) {
		vec4 result;
		float alfa = u / R2;
		float beta = v / R1;

		float x = (R1 + R2*cos(alfa))*cos(beta);
		float y = (R1 + R2*cos(alfa))*sin(beta);
		float z = R2*sin(alfa);
		result = vec4(x, y, z, 1);
		return result;
	}

	void AddPoint(float cX, float cY) {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		nVertices = 0;

		nPoints++;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();



		controlPoints.clear();

		if (nPoints == 1) {
			a = wVertex;
			controlPoints.push_back(a);
			loadPoints(controlPoints);
			drawPoint(a);
		}
		if (nPoints == 2) {
			b = wVertex;
			controlPoints.push_back(a);
			PAB = findP(a, b);
			controlPoints.push_back(PAB);
			controlPoints.push_back(b);
			loadPoints(controlPoints);

			drawPoint(a); drawPoint(b);  drawPoint(PAB);
		}
		if (nPoints == 3) {
			c = wVertex;

			controlPoints.push_back(a);
			controlPoints.push_back(PAB);
			controlPoints.push_back(b);
			loadPoints(controlPoints);
			controlPoints.clear();

			controlPoints.push_back(b);
			PBC = findP(b, c);
			controlPoints.push_back(PBC);
			controlPoints.push_back(c);
			loadPoints(controlPoints);
			controlPoints.clear();

			controlPoints.push_back(c);
			PCA = findP(c, a);
			controlPoints.push_back(PCA);
			controlPoints.push_back(a);
			loadPoints(controlPoints);

			drawPoint(a); drawPoint(b); drawPoint(c); drawPoint(PAB); drawPoint(PBC); drawPoint(PCA);

		}

		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 6 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}
	
	void drawPoint(vec4 v1) {
		for (float i = 0; i < 0.05; i+=0.01) {
			for (float j = 0; j < 0.05; j += 0.01) {
				vertexData[6 * nVertices] = v1.v[0]+i;
				vertexData[6 * nVertices + 1] = v1.v[1]+j;
				vertexData[6 * nVertices + 2] = 0;
				vertexData[6 * nVertices + 3] = 1;
				vertexData[6 * nVertices + 4] = 0;
				vertexData[6 * nVertices + 5] = 0;
				nVertices++;
			}
		}
		
	}

	vec4 findP(vec4 Point1, vec4 Point2) {
		float u1, v1, u2, v2;
		u1 = (Point1.v[0] + 10.f) / 20.f * 2 * M_PI*R2;
		u2 = (Point2.v[0] + 10.f) / 20.f * 2 * M_PI*R2;
		v1 = (Point1.v[1] + 10.f) / 20.f * 2 * M_PI*R1;
		v2 = (Point2.v[1] + 10.f) / 20.f * 2 * M_PI*R1;

		vector<vec4> tempCps;
		vec4 TempP;
		float minL=100000;
		vec4 bestP;
		float L;

		for (int i = 0; i < 100; i++) {
			float rand1 = (rand() % 2000) / 100.f - 10.f;
			float rand2 = (rand() % 2000) / 100.f - 10.f;
			TempP = vec4(rand1, rand2);

			tempCps.clear();
			tempCps.push_back(Point1);
			tempCps.push_back(TempP);
			tempCps.push_back(Point2);
			
			L = calculateDistance(tempCps);
			if (L < minL) {
				bestP = TempP;
				minL = L;
			}

		}

		tempCps.clear();
		tempCps.push_back(Point1);
		tempCps.push_back(bestP);
		tempCps.push_back(Point2);

		float test = calculateDistance(tempCps);
		printf("\n"); bestP.write();
		printf("\nminL=%f\n",  minL);
		
		return bestP;
	}
	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P()*camera.Rxyz();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_POINTS, 0, nVertices);
		}
	}
};

class GaussBg {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[300 * 200 * 6]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	GaussBg() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
									   // Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
																										// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));

		AddPoints();

	}


	void AddPoints() {
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		float k;
		float nPoint = 200.f;
		float density = 2 * M_PI / nPoint;
		int ui = 0;

		float minValue = 0;
		float maxValue = 0;
		for (float u = 0; u <= 2 * M_PI; u += density) {
			k = (1 / R2) * (cos(u) / (R1 + R2*cos(u)));

			if (k > maxValue)maxValue = k;
			if (k < minValue)minValue = k;


		}

		for (float u = 0; u <= 2 * M_PI; u += density) {

			k = (1 / R2) * (cos(u) / (R1 + R2*cos(u)));

			float green = 0;
			float red = 0;

			if (k < 0.f) {
				green = k / minValue;

			}
			else {
				red = k / maxValue;

			}

			for (float v = 0; v <= nPoint; v += 1) {


				float x = -10.f + (ui / (nPoint / 20));
				float y = 10.f - (v / (nPoint / 20));

				// fill interleaved data
				vertexData[6 * nVertices] = x;
				vertexData[6 * nVertices + 1] = y;
				vertexData[6 * nVertices + 2] = 0;
				vertexData[6 * nVertices + 3] = red;
				vertexData[6 * nVertices + 4] = green;
				vertexData[6 * nVertices + 5] = 0;


				nVertices++;
			}

			ui++;
		}

		printf("%d", nVertices);


		// copy data to the GPU
		glBufferData(GL_ARRAY_BUFFER, nVertices * 6 * sizeof(float), vertexData, GL_STATIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P()*camera.Rxyz();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_POINTS, 0, nVertices);
		}
	}
};
Torus torus;
Bezier bezier;
//GaussBg bg;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU

	torus.Create();
	bezier.Create();
	//bg.Create();

	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	bezier.Draw();
	//bg.Draw();
	torus.Draw();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'k') glutPostRedisplay();         // if d, invalidate display, i.e. redraw

	if (key == 'w') camera.a += 0.05;
	if (key == 's') camera.a -= 0.05;

	if (key == 'd') camera.b += 0.05;
	if (key == 'a') camera.b -= 0.05;

	if (key == 'q') camera.c += 0.05;
	if (key == 'e') camera.c -= 0.05;
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		bezier.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera

	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {

	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

