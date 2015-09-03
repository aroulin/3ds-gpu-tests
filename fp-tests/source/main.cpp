#include <stdio.h>
#include <3ds.h>

extern "C" {
#include "3dmath.h"
#include "gpu.h"
#include "vshader_shbin.h"
}

// Testing shader floating-point behavior
//
// Each test draws a full-screen quad, the color of which is determined by the
// vertex shader. The single shader contains several tests, selected by setting
// the `src1_uniform.x` uniform to specific values. For each test, the shader
// executes an instruction and then classifies the output to determine if it's
// NaN, +inf, -inf or a regular number. Based on this classification, the
// output color of the quad is set differently. It is then read by the test
// driver to compare against the expected result. This is used to check several
// attributes of the floating-point engine in the PICA200.
//
// See the shader file (vshader.pica) for specific details on each test.

// shader uniform src1_uniform
static vector_4f src1_uniform;
static int uLoc_src1_uniform;

struct vec3 {
	float x, y, z;
};

namespace Tests {

struct Testcase {
	int id;
	vec3 result;
	const char* description;
};

static const vec3 pinf = { 1.0f, 0.0f, 0.0f };
static const vec3 ninf = { 0.0f, 1.0f, 0.0f };
static const vec3 nan  = { 0.0f, 0.0f, 0.0f };
static const vec3 num  = { 1.0f, 0.0f, 1.0f }; // Not zero nor one
static const vec3 zero = { 1.0f, 1.0f, 0.0f };
static const vec3 one  = { 1.0f, 1.0f, 1.0f };

static const Testcase tests[] = {
	{ 0, pinf, "rcp(0) -> +inf"},
	{ 1, zero, "rcp(+inf) -> 0"},
	{ 2, nan,  "rcp(NaN) -> NaN"},
	{ 3, pinf, "rsq(0) -> +inf"},
	{ 4, one,  "rsq(1) -> 1"},
	{ 5, nan,  "rsq(-1) -> NaN"},
	{ 6, zero, "rsq(+inf) -> 0"},
	{ 7, nan,  "rsq(-inf) -> NaN"},
	{ 8, nan,  "rsq(NaN) -> NaN"},
	{ 9, pinf, "max(0, +inf) -> +inf"},
	{10, zero, "max(0, -inf) -> 0"},
	{11, nan,  "max(0, NaN) -> NaN"},
	{12, zero, "max(NaN, 0) -> 0"},
	{13, pinf, "max(-inf, +inf) -> +inf"},
	{14, zero, "min(0, +inf) -> 0"},
	{15, ninf, "min(0, -inf) -> -inf"},
	{16, nan,  "min(0, NaN) -> NaN"},
	{17, zero, "min(NaN, 0) -> 0"},
	{18, ninf, "min(-inf, +inf) -> -inf"},
	{19, nan,  "+inf - +inf -> NaN"},
	{20, zero, "+inf * 0 -> 0"},
	{21, zero, "0 * +inf -> 0"},
	{22, nan,  "NaN * 0 -> NaN"},
	{23, nan,  "0 * NaN -> NaN"},
	{24, one,  "mad(+inf, 0, 1) -> 1"},
	{25, num,  "dp4([...], [...]) -> 2"},
	{26, zero, "dp3([...], [...]) -> 0"},
	{27, one,  "dph([...], [...]) -> 1"},
	{28, zero, "sge(0, NaN) -> 0"},
	{29, zero, "sge(NaN, 0) -> 0"},
	{30, zero, "sgei(0, NaN) -> 0"},
	{31, zero, "sgei(NaN, 0) -> 0"},
	{32, zero, "slt(0, NaN) -> 0"},
	{33, zero, "slt(NaN, 0) -> 0"},
	{34, zero, "slti(0, NaN) -> 0"},
	{35, zero, "slti(NaN, 0) -> 0"},
	{36, one, "-flr(-0.1) -> 1"},
	{37, pinf, "rsq(rcp(-inf)) -> +inf"},
	{38, zero, "exp2(-inf) -> 0"},
	{39, ninf, "log2(rcp(-inf)) -> -inf"},
	{40, nan, "log2(-1) -> NaN"},
};

static size_t tests_count = (sizeof(tests)/sizeof(tests[0]));

}

//
// Testing framework boilerplate
//

static void sceneInit();
static void sceneRender();
static void sceneExit();
static void Verify(vec3 expected);

#define CLEAR_COLOR 0x0

int main() {
	// Initialize graphics
	gfxInitDefault();
	gpuInit();
	consoleInit(GFX_BOTTOM, NULL);

	// Initialize the scene
	sceneInit();

	printf("Press A to begin.\n");
	while(true) {
		gspWaitForVBlank();
		hidScanInput();
		if (hidKeysDown() & KEY_A)
			break;
	}

	// Run one test per frame
	using namespace Tests;

	for (size_t i = 0; i < tests_count; i++)
	{
		gpuClearBuffers(CLEAR_COLOR);

		printf("Test %d: %s\n", tests[i].id, tests[i].description);
		src1_uniform.x = (float)tests[i].id;

		gpuFrameBegin();
		sceneRender();
		gpuFrameEnd();

		Verify(tests[i].result);

		//Wait for the screen to be updated
		gfxSwapBuffersGpu();
		gspWaitForVBlank();
	}

	// End of tests
	printf("Tests ends. Press start to exit.\n");
	while(true) {
		gspWaitForVBlank();
		hidScanInput();
		if (hidKeysDown() & KEY_START)
			break;
	}

	// Deinitialize the scene
	sceneExit();

	// Deinitialize graphics
	gpuExit();
	gfxExit();
	return 0;
}

static void Verify(vec3 expected_result) {
	u8* framebuffer = gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL);
	GSPGPU_InvalidateDataCache(NULL, framebuffer, 3);

	u8 final_result[3];
	memcpy(final_result, framebuffer, 3);

	u8 expected[3] = {
		(u8)(expected_result.z * 255.0f),
		(u8)(expected_result.y * 255.0f),
		(u8)(expected_result.x * 255.0f),
	};

	if (expected[0] != final_result[0] || expected[1] != final_result[1] || expected[2] != final_result[2]) {
		printf("Failure: final=(%02X %02X %02X)\n"
		       "      expected=(%02X %02X %02X)\n",
			   (unsigned)final_result[2], (unsigned)final_result[1], (unsigned)final_result[0],
			   (unsigned)expected[2],     (unsigned)expected[1],     (unsigned)expected[0]);
		while(true) {
			gspWaitForVBlank();
			hidScanInput();
			if (hidKeysDown() & KEY_A)
				break;
		}
	}
}

typedef struct { float x, y, z; } vertex;

static vertex vertex_list[] =
{
		{ -1.0f, -1.0f, -0.5f },
		{  1.0f, -1.0f, -0.5f },
		{ -1.0f,  1.0f, -0.5f },
		{  1.0f,  1.0f, -0.5f },

};

static int vertex_list_count = sizeof(vertex_list)/sizeof(vertex_list[0]);

static DVLB_s* vshader_dvlb;
static shaderProgram_s program;

static void* vbo_data;

static void sceneInit()
{
	// Load the vertex shader and create a shader program
	vshader_dvlb = DVLB_ParseFile((u32*)vshader_shbin, vshader_shbin_size);
	shaderProgramInit(&program);
	shaderProgramSetVsh(&program, &vshader_dvlb->DVLE[0]);

	// Get the location of the projection matrix uniform
	uLoc_src1_uniform = shaderInstanceGetUniformLocation(program.vertexShader, "src1_uniform");

	// Create the VBO (vertex buffer object)
	vbo_data = linearAlloc(sizeof(vertex_list));
	memcpy(vbo_data, vertex_list, sizeof(vertex_list));

	GSPGPU_FlushDataCache(nullptr, (u8*)vbo_data, sizeof(vertex_list));
}

static void sceneRender()
{
	// Bind the shader program
	shaderProgramUse(&program);

	// Configure the first fragment shading substage to just pass through the vertex color
	// See https://www.opengl.org/sdk/docs/man2/xhtml/glTexEnv.xml for more insight
	GPU_SetTexEnv(0,
				  GPU_TEVSOURCES(GPU_PRIMARY_COLOR, 0, 0), // RGB channels
				  GPU_TEVSOURCES(GPU_PRIMARY_COLOR, 0, 0), // Alpha
				  GPU_TEVOPERANDS(0, 0, 0), // RGB
				  GPU_TEVOPERANDS(0, 0, 0), // Alpha
				  GPU_REPLACE, GPU_REPLACE, // RGB, Alpha
				  0xFFFFFFFF);

	// Configure the "attribute buffers" (that is, the vertex input buffers)
	u32 buffer_offsets[1] = { 0x0 };
	u64 attribute_map[1] = { 0x0 };
	u8 num_attributes[1] = { 1 };
	GPU_SetAttributeBuffers(
			1, // Number of inputs per vertex
			(u32*)osConvertVirtToPhys((u32)vbo_data), // Location of the VBO
			GPU_ATTRIBFMT(0, 3, GPU_FLOAT), // Format of the inputs (in this case the only input is a 3-element float vector)
			0xFFE, // Unused attribute mask, in our case bit 0 is cleared since it is used
			0x0, // Attribute permutations (here it is the identity)
			1, // Number of buffers
			buffer_offsets, // Buffer offsets (placeholders)
			attribute_map, // Attribute permutations for each buffer (identity again)
			num_attributes); // Number of attributes for each buffer

	// Upload the test vector uniform
	GPU_SetFloatUniform(GPU_VERTEX_SHADER, (u32) uLoc_src1_uniform, (u32*)&src1_uniform, 1);

	// Draw the VBO
	GPU_DrawArray(GPU_TRIANGLE_STRIP, vertex_list_count);
}

static void sceneExit()
{
	// Free the VBO
	linearFree(vbo_data);

	// Free the shader program
	shaderProgramFree(&program);
	DVLB_Free(vshader_dvlb);
}
