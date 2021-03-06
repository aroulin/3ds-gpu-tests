#include <stdio.h>
#include <3ds/gpu/shaderProgram.h>

#include "3dmath.h"
#include "gpu.h"
#include "test.h"
#include "vshader_shbin.h"

// Testing the DPH shader instruction

// For each test:
// Draws a 1px by 1px quad in the bottom-left corner using a vertex shader
// The DPH instruction is used to output the color of this quad.
// The framebuffer is then read to verify the results

// The exact DPH shader command under test:
// dph outclr.xyz, src1_uniform, src2_in_color

// shader input color
struct { float r, g, b, a; } src2_in_color;

// shader uniform src1_uniform
static vector_4f src1_uniform;
static int uLoc_src1_uniform;

// Expected result
static vector_4f expected_result;

static void Test_DPH_Zeros(void) {
	printf("Test: DPH_Zeros\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 0.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 0.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 0.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.0f;
}

static void Test_DPH_Zeros2(void) {
	printf("Test: DPH_Zeros\n");
	src1_uniform.x = 1.0f, src2_in_color.r = 0.0f, expected_result.x = 0.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 1.0f, expected_result.y = 0.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 1.0f, expected_result.z = 0.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.0f;
}

static void Test_DPH_X(void) {
	printf("Test: DPH_X\n");
	src1_uniform.x = 1.0f, src2_in_color.r = 1.0f, expected_result.x = 1.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 1.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 1.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.0f;
}

static void Test_DPH_Y(void) {
	printf("Test: DPH_Y\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 1.0f;
	src1_uniform.y = 1.0f, src2_in_color.g = 1.0f, expected_result.y = 1.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 1.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.0f;
}

static void Test_DPH_Z(void) {
	printf("Test: DPH_Z\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 1.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 1.0f;
	src1_uniform.z = 1.0f, src2_in_color.b = 1.0f, expected_result.z = 1.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.0f;
}

static void Test_DPH_W(void) {
	printf("Test: DPH_W\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 1.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 1.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 1.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 1.0f;
}

static void Test_DPH_W2(void) {
	printf("Test: DPH_W2\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 0.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 0.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 0.0f;
	src1_uniform.w = 1.0f, src2_in_color.a = 0.0f;
}

static void Test_DPH_Simple(void) {
	printf("Test: DPH_Simple\n");
	src1_uniform.x = 0.5f, src2_in_color.r = 0.5f, expected_result.x = 1.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 1.0f;
	src1_uniform.z = 0.5f, src2_in_color.b = 0.5f, expected_result.z = 1.0f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.5f;
}

static void Test_DPH_Simple2(void) {
	printf("Test: DPH_Simple2\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 0.75f;
	src1_uniform.y = 0.5f, src2_in_color.g = 0.5f, expected_result.y = 0.75f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 0.75f;
	src1_uniform.w = 0.0f, src2_in_color.a = 0.5f;
}

static void Test_DPH_Simple3(void) {
	printf("Test: DPH_Simple3\n");
	src1_uniform.x = 0.0f, src2_in_color.r = 0.0f, expected_result.x = 1.0f;
	src1_uniform.y = 0.0f, src2_in_color.g = 0.0f, expected_result.y = 1.0f;
	src1_uniform.z = 0.0f, src2_in_color.b = 0.0f, expected_result.z = 1.0f;
	src1_uniform.w = 0.5f, src2_in_color.a = 1.0f;
}

static test_t tests[] = {
	&Test_DPH_Zeros,
	&Test_DPH_Zeros2,
	&Test_DPH_X,
	&Test_DPH_Y,
	&Test_DPH_Z,
	&Test_DPH_W,
	&Test_DPH_W2,
	&Test_DPH_Simple,
	&Test_DPH_Simple2,
	&Test_DPH_Simple3,
};

static int tests_count =  (sizeof(tests)/sizeof(tests[0]));

//
// Testing framework boilerplate
//

static void sceneInit(void);
static void sceneRender(void);
static void sceneExit(void);
static void Verify(void);

#define CLEAR_COLOR 0x0

int main(void)
{
	// Initialize graphics
	gfxInitDefault();
	gpuInit();
	consoleInit(GFX_BOTTOM, NULL);

	// Initialize the scene
	sceneInit();
	gpuClearBuffers(CLEAR_COLOR);

	// Run one test per frame
	for (int i = 0; i < tests_count; i++)
	{
		tests[i]();
			gpuFrameBegin();
				sceneRender();
			gpuFrameEnd();
		Verify();

		gpuClearBuffers(CLEAR_COLOR);
		gspWaitForVBlank();  // Synchronize with the start of VBlank
		gfxSwapBuffersGpu(); // Swap the framebuffers so that the frame that we rendered last frame is now visible

		// Flush the framebuffers out of the data cache (not necessary with pure GPU rendering)
		//gfxFlushBuffers();
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

static void Verify(void) {
	u32 final_result = ((unsigned*)gfxGetFramebuffer(GFX_TOP, GFX_LEFT, NULL, NULL))[0];
	if ((unsigned)(expected_result.z * 255.0f) != (final_result & 0xFF))
		printf("Failure final.z = %x\n, expected = %x\n", (unsigned)(final_result & 0xFF), (unsigned)(expected_result.z*255.0f));
	if ((unsigned)(expected_result.y * 255.0f) != ((final_result >> 8) & 0xFF))
		printf("Failure final.y = %x\n, expected = %x\n", (unsigned)((final_result >> 8) & 0xFF), (unsigned)(expected_result.y*255.0f));
	if ((unsigned)(expected_result.x * 255.0f) != ((final_result >> 16) & 0xFF))
		printf("Failure final.x = %x\n, expected = %x\n", (unsigned)((final_result >> 16) & 0xFF), (unsigned)(expected_result.x*255.0f));
}

typedef struct { float x, y, z; float r, g, b, a; } vertex;

static vertex vertex_list[] =
{
		{ 0.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f },
		{ 1.0f, 0.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f },
		{ 1.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f },
		{ 0.0f, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f },

};

static int vertex_list_count = sizeof(vertex_list)/sizeof(vertex_list[0]);

static DVLB_s* vshader_dvlb;
static shaderProgram_s program;
static int uLoc_projection;
static matrix_4x4 projection;

static void* vbo_data;

static void sceneInit(void)
{
	// Load the vertex shader and create a shader program
	vshader_dvlb = DVLB_ParseFile((u32*)vshader_shbin, vshader_shbin_size);
	shaderProgramInit(&program);
	shaderProgramSetVsh(&program, &vshader_dvlb->DVLE[0]);

	// Get the location of the projection matrix uniform
	uLoc_projection = shaderInstanceGetUniformLocation(program.vertexShader, "projection");
	uLoc_src1_uniform = shaderInstanceGetUniformLocation(program.vertexShader, "src1_uniform");

	// Compute the projection matrix
	m4x4_ortho_tilt(&projection, 0.0, 400.0, 0.0, 240.0, 0.0, 1.0);

	// Create the VBO (vertex buffer object)
	vbo_data = linearAlloc(sizeof(vertex_list));
	memcpy(vbo_data, vertex_list, sizeof(vertex_list));
}

static void sceneRender(void)
{
	for (int i = 0; i < vertex_list_count; i++) {
		vertex_list[i].r = src2_in_color.r;
		vertex_list[i].g = src2_in_color.g;
		vertex_list[i].b = src2_in_color.b;
		vertex_list[i].a = src2_in_color.a;
	}
	memcpy(vbo_data, vertex_list, sizeof(vertex_list));

	// Bind the shader program
	shaderProgramUse(&program);

	// Configure the first fragment shading substage to just pass through the vertex color
	// See https://www.opengl.org/sdk/docs/man2/xhtml/glTexEnv.xml for more insight
	GPU_SetTexEnv(0,
				  GPU_TEVSOURCES(GPU_PRIMARY_COLOR, GPU_PRIMARY_COLOR, GPU_PRIMARY_COLOR), // RGB channels
				  GPU_TEVSOURCES(GPU_PRIMARY_COLOR, GPU_PRIMARY_COLOR, GPU_PRIMARY_COLOR), // Alpha
				  GPU_TEVOPERANDS(0, 0, 0), // RGB
				  GPU_TEVOPERANDS(0, 0, 0), // Alpha
				  GPU_REPLACE, GPU_REPLACE, // RGB, Alpha
				  0xFFFFFFFF);

	// Configure the "attribute buffers" (that is, the vertex input buffers)
	GPU_SetAttributeBuffers(
			2, // Number of inputs per vertex
			(u32*)osConvertVirtToPhys((u32)vbo_data), // Location of the VBO
			GPU_ATTRIBFMT(0, 3, GPU_FLOAT) | GPU_ATTRIBFMT(1, 4, GPU_FLOAT), // Format of the inputs (in this case the only input is a 3-element float vector)
			0xFFC, // Unused attribute mask, in our case bit 0 is cleared since it is used
			0x10, // Attribute permutations (here it is the identity)
			1, // Number of buffers
			(u32[]) { 0x0 }, // Buffer offsets (placeholders)
			(u64[]) { 0x10 }, // Attribute permutations for each buffer (identity again)
			(u8[])  { 2 }); // Number of attributes for each buffer

	// Upload the projection matrix
	GPU_SetFloatUniformMatrix(GPU_VERTEX_SHADER, uLoc_projection, &projection);

	// Upload the test vector uniform
	GPU_SetFloatUniform(GPU_VERTEX_SHADER, (u32) uLoc_src1_uniform, (u32*)&src1_uniform, 1);

	// Draw the VBO
	GPU_DrawArray(GPU_TRIANGLE_STRIP, vertex_list_count);
}

static void sceneExit(void)
{
	// Free the VBO
	linearFree(vbo_data);

	// Free the shader program
	shaderProgramFree(&program);
	DVLB_Free(vshader_dvlb);
}
