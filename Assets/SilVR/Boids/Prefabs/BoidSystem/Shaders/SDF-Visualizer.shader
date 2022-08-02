Shader "SilVR/Nave-Raymarching/SDF-Visualizer"
{
	Properties
	{
		_AO("Ambient Occlusion", Range(0, 5)) = 1.0
		_Color("Color", Color) = (1,1,1,1)
		_Reflect("Reflectivity", Range(0,1)) = 0
		_Cubemap("Cubemap", Cube) = "black" {}
		_SdfScale("SDF Scale", float) = 1

		_Vector("SDF testing vector", Vector) = (0,0,0,0)
		_Float0("SDF testing float 0", float) = 1
		_Float1("SDF testing float 1", float) = 1

		_Debug("Debug", float) = 1
	}
	SubShader
	{
		Tags { "RenderType"="Transparent" "Queue"="Transparent-1" }
		LOD 100
		Cull Front
		ZWrite Off
		Blend SrcAlpha OneMinusSrcAlpha

		Pass
		{
			CGPROGRAM
			#pragma vertex vert
			#pragma fragment frag
			
			#include "UnityCG.cginc"
			
			#define VISUALIZER 1
			#include "Assets\SilVR\Boids\Prefabs\BoidSystem\Shaders\Includes\silvr-SDF0.cginc"

			#define PI 3.14159265
			#define ROOT_HALF .707106781
			struct appdata
			{
				float4 vertex : POSITION;
			};

			struct v2f
			{
				float4 vertex : SV_POSITION;
				float3 wpos : TEXCOORD0;
				float3 wcen : TEXCOORD1;
			};

			struct fragOut
			{
				float4 color : SV_Target;
				float depth : SV_Depth;
			};
			
			sampler2D_float _CameraDepthTexture;
			samplerCUBE _Cubemap;
			float _AO;
			float _Reflect;
			fixed4 _Color;
			float _SdfScale;
			float _Debug;

			float4 _Vector;
			float _Float0;
			float _Float1;

			v2f vert (appdata v)
			{
				v2f o;
				o.vertex = UnityObjectToClipPos(v.vertex);
				o.wpos = mul(unity_ObjectToWorld, v.vertex).xyz;
				o.wcen = mul(unity_ObjectToWorld, float4(0, 0, 0, 1));
				return o;
			}
			
			// Simple raymarcher from Nave and Neen

			fragOut frag (v2f i)
			{
				//Heyy Neen, I'm taking the time to comment this for you :)
				fragOut f;

				//Alright so this just defines the maximum amount of steps we'll use, I probably don't
				//have to explain something as simple as this but whatever :D
				float maxStep = 64;
				float4 clipPos = float4(0,0,0,0);
				float clipDepth = -1;
				float3 normal = float3(0, 1, 0);

				fixed4 colorHit = float4(0,0,0,1);

				//alright, let's get straight into raycasting, getting the direction and origin of the
				//ray is super easy, for the direction you just take the vector between the camera position
				//and the world position of the current pixel that's processed and tada, you got the direction
				//normalize this shit and you're good to go
				float3 raydir = normalize(i.wpos - _WorldSpaceCameraPos.xyz);
				//the position the ray starts is just as easy, it's our camera position, who would have guessed that
				float3 raypos = (_WorldSpaceCameraPos.xyz - i.wcen) * _SdfScale;
				
				float flipped = 1;

				float d = 1000;
				//alright, time to raytrace, simple for-loop with maxStep iterations
				for (int stp = 0; stp < maxStep; stp++) {
					//alright, here we simply get the distance to the nearest object calculated by our amazing DE
					//or "Distance Estimation" function, genius, you can look into the comments on that one further above
					d = DE(raypos);
					//now if the distance is super small, that means we hit something, I tried checking against <= 0.0 but
					//that made everything noisy and stuff so we'll just use something super tiny here
					if (d < 0)
					{
						break;
					}
					//if (d <= 0.008 + .04*stp*.015625) {
					if (d <= 0.008) {
						//Now if we did hit something, we just return white times my magic ambient occlusion formular... Ohhhh :O
						//it's super simple tho, the main core is (stp / maxStep) which basically gives us the ratio of how many
						//steps it took to get here, if we hit something on the first step, then stp/maxStep is gonna be super small
						//but if it was the last step of the for loop then stp/maxStep is basically almost 1...
						//so if we hit something early it's gonna be close to 0 and if it's super far away or we needed a lot of steps
						//it's gonna be close to 1... then we just invert that with a "1 - bla" so, far is 1 and near is 0 and then
						//we take all of that and take that to a power of something which has a slider so you can kinda play around with
						//the "intensity" a little bit... Oh yes, also, if a ray takes multiple steps to get somewhere, not only does that
						//mean it may be far away, but it could also mean the surface it hit was more complex to get to, that's why you
						//see the spheres having some small gradient torwards the edges, which looks cool!
						clipPos = mul(UNITY_MATRIX_VP, float4(raypos*(1.0 / _SdfScale) + i.wcen, 1.0));
						clipDepth = clipPos.z / clipPos.w;
		
						normal = calcNormal(raypos, flipped);
						break;
					}
					
					//oh yes, also, if we didn't hit something we just add the direction times our minimum distance to the nearest
					//object to the current ray pos and take the next step
					raypos += raydir * d;
				}
				
				//also look, if it went through all steps and didn't hit anything then just return pure black... it must have either
				//went on for infinity or found itself in a super complex crack of some fractal surface or whatever
				colorHit = texCUBE(_Cubemap, reflect(raydir, normal));
				float4 reflective = colorHit * float4(1, 1, 1, 1) * saturate(pow(1 - stp / maxStep, _AO))*_Color * 4;
				
				float4 matte = float4(1, 1, 1, 1) * saturate(1 - stp / maxStep)*_Color * 4;
				
				float fx = (frac(raypos.x) - 0.5);
				float fy = (frac(raypos.y) - 0.5);
				float fz = (frac(raypos.z) - 0.5);

				f.color = float4( (matte * (1.25 + flipped) * .2).xyz, 1.0 ) * float4(0.5 + 0.25*fx - 0.25*fy, 0.5 + fy*0.25 - fz*0.25, 0.5 + fz*0.25 - fx*0.25, _Color.a);
				f.depth = clipDepth;
				return f;
			}
			ENDCG
		}
	}
}
