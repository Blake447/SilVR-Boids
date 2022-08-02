using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using System.IO;

namespace SilVR
{

    class BoidGenerateAssets : EditorWindow
    {
        string myString = "Hello World";
        bool groupEnabled;
        bool myBool = true;
        float myFloat = 1.23f;

        bool autoAssign = true;

        int texture_width = 32;
        int texture_height = 32;
        int boid_count = 1024;
        GameObject boidPrefab = null;

        Shader baseShaderSim = null;
        Shader baseShaderBoid = null;

        Material baseSimulatorMaterial = null;
        Material baseBoidMaterial = null;

        bool copySettingsFromMaterial = true;

        Shader newSimShader = null;
        Shader newBoidShader = null;

        bool generateNewBoidShader = false;

        string boidSetName = "fish";
        string meshSubname = "main";
        string simShaderSubname = "main";
        string boidShaderSubname = "main";
        string renderTextureSubname = "main";

        Mesh boidBaseMesh = null;

        [MenuItem("SilVR/Boids/Generate Assets")]

        public static void ShowWindow()
        {
            EditorWindow.GetWindow(typeof(BoidGenerateAssets));
        }

        private Mesh GenerateBoidMesh(Mesh baseMesh)
        {
            Mesh boidMesh = new Mesh();
            boidMesh.indexFormat = UnityEngine.Rendering.IndexFormat.UInt32;

            int tri_count = boidBaseMesh.triangles.Length * boid_count;
            int vert_count = boidBaseMesh.triangles.Length * boid_count;

            Vector3[] vertices = new Vector3[vert_count];
            int[] triangles = new int[tri_count];
            Vector2[] uvs = new Vector2[vert_count];
            Vector2[] datapacked_uvs = new Vector2[vert_count];
            Color[] colors = new Color[vert_count];

            Debug.Log("Base mesh vertex count: " + boidBaseMesh.vertices.Length);
            Debug.Log("Base uv vertex count: " + boidBaseMesh.uv.Length);

            Vector3[] base_vertices = boidBaseMesh.vertices;
            int[] base_triangles = boidBaseMesh.triangles;
            Vector2[] base_uvs = boidBaseMesh.uv;
            //Color[] base_colors = boidBaseMesh.colors;

            for (int iBoid = 0; iBoid < boid_count; iBoid++)
            {
                int xPixelIndex = iBoid % texture_width;
                int yPixelIndex = iBoid / texture_width;

                float xPixelCenterCoordinate = (xPixelIndex + 0.5f) / texture_width;
                float yPixelCenterCoordinate = (yPixelIndex + 0.5f) / texture_height;

                Vector2 pixelCenterCoordinate = new Vector2(xPixelCenterCoordinate, yPixelCenterCoordinate);

                int originalVertexCount = base_vertices.Length;
                int originalTriangleCount = base_triangles.Length;
                for (int jVertex = 0; jVertex < base_vertices.Length; jVertex++)
                {
                    int vertexIndexOffset = originalVertexCount * iBoid;
                    vertices[vertexIndexOffset + jVertex] = base_vertices[jVertex];
                    //colors[vertexIndexOffset + jVertex] = base_colors[jVertex];
                    uvs[vertexIndexOffset + jVertex] = base_uvs[jVertex];
                    datapacked_uvs[vertexIndexOffset + jVertex] = pixelCenterCoordinate;
                }
                for (int jTri = 0; jTri < base_triangles.Length; jTri++)
                {
                    int triangleIndexOffset = originalTriangleCount * iBoid;
                    int vertexIndexOffset = originalVertexCount * iBoid;

                    triangles[triangleIndexOffset + jTri] = base_triangles[jTri] + vertexIndexOffset;
                }
            }

            boidMesh.vertices = vertices;
            boidMesh.triangles = triangles;
            boidMesh.uv = uvs;
            boidMesh.uv2 = datapacked_uvs;
            //boidMesh.colors = colors;
            return boidMesh;

        }
        private void ApplyMeshToPrefab(GameObject prefabRootObject, Mesh mesh)
        {
            int SIMULATOR_OFFSET = 0;
            int BOIDS_ROOT = 1;

            GameObject boidsRoot = prefabRootObject.transform.GetChild(BOIDS_ROOT).gameObject;
            int BOID_MESH_INDEX = 0;
            GameObject MeshBoidsObj = boidsRoot.transform.GetChild(BOID_MESH_INDEX).gameObject;
            SkinnedMeshRenderer mr = MeshBoidsObj.GetComponent<SkinnedMeshRenderer>();
            MeshFilter meshFilter = MeshBoidsObj.GetComponent<MeshFilter>();
            mr.sharedMesh = mesh;
            meshFilter.sharedMesh = mesh;
        }
        void FindReplaceCharArray(char[] charArray, string template, string replacement)
        {
            string variantName = replacement;
            string templateName = template;
            while (variantName.Length < templateName.Length)
            {
                variantName += " ";
            }

            int charSentinel = 0;
            int start = 0;
            for (int i = 0; i < charArray.Length; i++)
            {
                if (charArray[i] == templateName[charSentinel])
                {
                    if (charSentinel == 0)
                    {
                        start = i;
                    }
                    charSentinel++;
                }
                else
                {
                    charSentinel = 0;
                }

                if (charSentinel >= templateName.Length)
                {
                    for (int j = 0; j < templateName.Length; j++)
                    {
                        charArray[start + j] = variantName[j];
                    }
                    break;
                }
            }
        }
        private Shader GenerateSimulatorShaderAsset(Shader baseShader, string path, string name, int width, int height)
        {
            //int width = 32;
            //int height = 32;
            string res = width + "x" + height + "-";
            string label = res + name;


            string simShaderFilePath = path;
            using (StreamReader shader_reader = new StreamReader(AssetDatabase.GetAssetPath(baseShader)))
            {
                Debug.Log(AssetDatabase.GetAssetPath(baseShader));
                string file_content = shader_reader.ReadToEnd();

                char[] file_out = @file_content.ToCharArray();

                string templateName = "32x32_TemplateShaderFile";
                string variantName = label;
                FindReplaceCharArray(file_out, templateName, variantName);

                string templateWidth = "#define TEXTURE_WIDTH 0032.0";
                string templateHeight = "#define TEXTURE_HEIGHT 0032.0";

                string defineWidth = "#define TEXTURE_WIDTH " + width.ToString("0000") + ".0";
                string defineHeight = "#define TEXTURE_HEIGHT " + width.ToString("0000") + ".0";

                FindReplaceCharArray(file_out, templateWidth, defineWidth);
                FindReplaceCharArray(file_out, templateHeight, defineHeight);

                string file_out_string = new string(file_out);
                File.WriteAllText(simShaderFilePath, @file_out_string);
                shader_reader.Close();
                AssetDatabase.Refresh();
            }
            return AssetDatabase.LoadAssetAtPath<Shader>(simShaderFilePath);
        }
        private void ApplySimulatorToPrefab(GameObject prefabRootObject, Material simulatorMaterial)
        {
            int SIMULATOR_OFFSET = 0;
            int BOIDS_ROOT = 1;

            GameObject particleSimulator = prefabRootObject.transform.GetChild(SIMULATOR_OFFSET).gameObject;
            GameObject simulationPlane = particleSimulator;
            MeshRenderer simPlaneRenderer = simulationPlane.GetComponent<MeshRenderer>();
            simPlaneRenderer.sharedMaterial = simulatorMaterial;
        }
        private Shader GenerateBoidShaderAsset(Shader baseShader, string path)
        {
            string boidShaderFilePath = path;
            StreamWriter sw0 = File.CreateText(boidShaderFilePath);
            using (StreamReader shader_reader0 = new StreamReader(AssetDatabase.GetAssetPath(baseShader)))
            {
                string file_content0 = shader_reader0.ReadToEnd();
                sw0.Write(file_content0);
                sw0.Flush();
                shader_reader0.Close();
                sw0.Close();
                AssetDatabase.Refresh();
            }
            return AssetDatabase.LoadAssetAtPath<Shader>(boidShaderFilePath);
        }
        private void ApplyBoidMaterialToPrefab(GameObject prefabRootObject, Material boidMaterial)
        {
            int SIMULATOR_OFFSET = 0;
            int BOIDS_ROOT = 1;

            GameObject boidsRoot = prefabRootObject.transform.GetChild(BOIDS_ROOT).gameObject;
            int BOID_MESH_INDEX = 0;
            GameObject MeshBoidsObj = boidsRoot.transform.GetChild(BOID_MESH_INDEX).gameObject;
            SkinnedMeshRenderer mr = MeshBoidsObj.GetComponent<SkinnedMeshRenderer>();
            mr.sharedMaterial = boidMaterial;
        }
        private void ApplyRenderTextureToPrefab(GameObject prefabRootObject, RenderTexture renderTexture)
        {
            int SIMULATOR_OFFSET = 0;
            int BOIDS_ROOT = 1;

            int P_NOISE_TRANSFORM = 0;
            int P_CAMERA_TRANSFORM = 1;


            GameObject particleSimulator = prefabRootObject.transform.GetChild(SIMULATOR_OFFSET).gameObject;
            GameObject boidsRoot = prefabRootObject.transform.GetChild(BOIDS_ROOT).gameObject;

            GameObject noiseToggle = particleSimulator.transform.GetChild(P_NOISE_TRANSFORM).gameObject;
            GameObject cameraTransform = particleSimulator.transform.GetChild(P_CAMERA_TRANSFORM).gameObject;

            GameObject cameraObj = cameraTransform.transform.GetChild(0).gameObject;
            Debug.Log("Camera object name: " + cameraObj.name);
            Camera camera = cameraObj.GetComponent<Camera>();
            if (camera == null)
            {
                Debug.LogError("Could not find camera component");
            }

            int BOID_MESH_INDEX = 0;

            GameObject MeshBoidsObj = boidsRoot.transform.GetChild(BOID_MESH_INDEX).gameObject;
            SkinnedMeshRenderer mr = MeshBoidsObj.GetComponent<SkinnedMeshRenderer>();
            MeshFilter meshFilter = MeshBoidsObj.GetComponent<MeshFilter>();

            GameObject simulationPlane = particleSimulator;
            MeshRenderer simPlaneRenderer = simulationPlane.GetComponent<MeshRenderer>();

            simPlaneRenderer.sharedMaterial.SetTexture("_MainTex", renderTexture);
            mr.sharedMaterial.SetTexture("_MainTex", renderTexture);
            camera.targetTexture = renderTexture;
        }
        private Material GenerateMaterial(Shader sourceShader)
        {
            Material newMaterial = new Material(sourceShader);
            return newMaterial;
        }
        private Material GenerateMaterial(Shader sourceShader, Material baseMaterial)
        {
            Material newMaterial = new Material(baseMaterial);
            newMaterial.shader = sourceShader;
            return newMaterial;
        }


        private void OnGUI()
        {
            string outputPath = "Assets/SilVR/Boids/Output/";
            string defaultSourceSimulatorShader = "Assets/SilVR/Boids/Prefabs/BoidSystem/Shaders/silvr-boid-simulator-template.shader";
            string defaultSourceMeshboidShader = "Assets/SilVR/Boids/Prefabs/BoidSystem/Shaders/silvr-boid-meshboids-template.shader";



            string simulationShaderPath = "Assets/SilVR/Boids/Shaders/silvr-boid-simulator-32x32.shader";
            string noiseTexturePath = "Assets/SilVR/Boids/Textures/colorNoise.jpg";
            string boidShaderPath = "Assets/SilVR/Boids/Shaders/silvr-boid-meshboids-32x32.shader";
            string prefabPath = "Assets/SilVR/Boids/Prefabs/BoidSystem.prefab";


            //autoAssign = EditorGUILayout.Toggle("Auto Assign Assets", autoAssign);

            GUILayout.Label("Prefab to Apply Changes To");
            boidPrefab = (GameObject)EditorGUILayout.ObjectField("Existing boid prefab", boidPrefab, typeof(GameObject), true);


            GUILayout.Space(8.0f);
            GUILayout.Label("Boid Set Properties", EditorStyles.boldLabel);
            boidSetName = (string)EditorGUILayout.TextField("Boid set name", boidSetName);
            texture_width = (int)EditorGUILayout.IntField("Texture width", texture_width);
            texture_height = (int)EditorGUILayout.IntField("Texture height", texture_height);
            boid_count = (int)EditorGUILayout.IntField("Boid Count", boid_count);

            boidBaseMesh = (Mesh)EditorGUILayout.ObjectField("Boid base mesh", boidBaseMesh, typeof(Mesh), true);

            //baseShaderSim = (Shader)EditorGUILayout.ObjectField("Base Sim Shader", baseShaderSim, typeof(Shader), true);
            //baseShaderBoid = (Shader)EditorGUILayout.ObjectField("Base Boid Shader", baseShaderBoid, typeof(Shader), true);

            GUILayout.Space(8.0f);
            GUILayout.Label("Base Assets to Generate From", EditorStyles.boldLabel);
            baseBoidMaterial = (Material)EditorGUILayout.ObjectField("Base Boid Material", baseBoidMaterial, typeof(Material), true);
            baseSimulatorMaterial = (Material)EditorGUILayout.ObjectField("Base Simulator Material", baseSimulatorMaterial, typeof(Material), true);
            copySettingsFromMaterial = EditorGUILayout.Toggle("Copy Settings From Material", copySettingsFromMaterial);

            GUILayout.Space(8.0f);
            GUILayout.Label("Subname Overrides", EditorStyles.boldLabel);
            meshSubname = EditorGUILayout.TextField("Mesh Subname", meshSubname);
            simShaderSubname = EditorGUILayout.TextField("Sim Shader Subname", simShaderSubname);
            boidShaderSubname = EditorGUILayout.TextField("Boid Shader Subname", boidShaderSubname);
            renderTextureSubname = EditorGUILayout.TextField("RenderTexture Subname", renderTextureSubname);


            GUILayout.Space(4.0f);

            GUILayout.Label("Generate Assets", EditorStyles.boldLabel);

            // TODO: Factor individual buttons into own methods, so this isnt copy-pasted
            if (GUILayout.Button("Generate All Assets"))
            {
                bool applyToPrefab = autoAssign && boidPrefab != null;
                string resolution_label = texture_width + "x" + texture_height;

                if (texture_width * texture_height != boid_count)
                {
                    Debug.LogWarning("Warning: number of boids do not fit exactly in supplied texture resolution");
                }
                if (boidBaseMesh != null)
                {
                    Mesh boidMesh = GenerateBoidMesh(boidBaseMesh);
                    string boidMeshName = boidSetName + "-" + meshSubname + "-boids-mesh-" + resolution_label + ".mesh";
                    AssetDatabase.CreateAsset(boidMesh, outputPath + boidMeshName);
                    if (applyToPrefab)
                    {
                        ApplyMeshToPrefab(boidPrefab, boidMesh);
                    }
                }
                baseShaderSim = (Shader)AssetDatabase.LoadAssetAtPath(defaultSourceSimulatorShader, typeof(Shader));
                if (baseShaderSim != null)
                {
                    string simShaderFilePath = outputPath + boidSetName + "-" + simShaderSubname + "-sim-shader-" + resolution_label + ".shader";
                    string shaderName = boidSetName + "-" + simShaderSubname;
                    newSimShader = GenerateSimulatorShaderAsset(baseShaderSim, simShaderFilePath, shaderName, texture_width, texture_height);

                    bool useBaseMaterial = copySettingsFromMaterial && baseSimulatorMaterial != null;
                    Material generatedSimulatorMaterial = useBaseMaterial ? GenerateMaterial(newSimShader, baseSimulatorMaterial) : GenerateMaterial(newSimShader);

                    string simMaterialName = boidSetName + "-" + simShaderSubname + "-sim-mat-" + resolution_label + ".mat";
                    string simMaterialFilePath = outputPath + simMaterialName;
                    AssetDatabase.CreateAsset(generatedSimulatorMaterial, simMaterialFilePath);
                    if (applyToPrefab)
                    {
                        ApplySimulatorToPrefab(boidPrefab, generatedSimulatorMaterial);
                    }
                }
                else
                {
                    Debug.LogError("Failed to load template simulator shader. Please ensure it has not been moved or renamed.");
                    Debug.LogError("Expected file path: " + defaultSourceSimulatorShader);
                }
                baseShaderBoid = (Shader)AssetDatabase.LoadAssetAtPath(defaultSourceMeshboidShader, typeof(Shader));
                if (baseShaderBoid != null)
                {
                    string boidShaderName = boidSetName + "-" + boidShaderSubname + "-boid-shader-" + resolution_label + ".shader";
                    string boidShaderFilePath = outputPath + boidShaderName;
                    //newBoidShader = GenerateBoidShaderAsset(baseShaderBoid, boidShaderFilePath);
                    newBoidShader = baseShaderBoid;

                    bool useBaseMaterial = copySettingsFromMaterial && baseBoidMaterial != null;
                    Material generatedBoidMaterial = useBaseMaterial ? GenerateMaterial(newBoidShader, baseBoidMaterial) : GenerateMaterial(newBoidShader);

                    string simMaterialName = boidSetName + "-" + boidShaderSubname + "-boid-mat-" + resolution_label + ".mat";
                    string simMaterialFilePath = outputPath + simMaterialName;
                    AssetDatabase.CreateAsset(generatedBoidMaterial, simMaterialFilePath);

                    if (applyToPrefab)
                    {
                        ApplyBoidMaterialToPrefab(boidPrefab, generatedBoidMaterial);
                    }
                }
                else
                {
                    Debug.LogError("Failed to load template boid shader. Please ensure it has not been moved or renamed.");
                    Debug.LogError("Expected file path: " + defaultSourceMeshboidShader);
                }



                RenderTexture renderTexture = new RenderTexture(texture_width * 2, texture_height * 4, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                renderTexture.filterMode = FilterMode.Point;
                renderTexture.anisoLevel = 0;
                AssetDatabase.CreateAsset(renderTexture, outputPath + boidSetName + "-" + renderTextureSubname + "-rtex-" + resolution_label + ".renderTexture");
                if (applyToPrefab)
                {
                    ApplyRenderTextureToPrefab(boidPrefab, renderTexture);
                }
            }

            GUILayout.Space(12.0f);
            GUILayout.Label("Advanced: Generate Individual Assets", EditorStyles.boldLabel);

            if (GUILayout.Button("Generate Mesh"))
            {
                if (texture_width * texture_height != boid_count)
                {
                    Debug.LogWarning("Warning: number of boids do not fit exactly in supplied texture resolution");
                }
                if (boidBaseMesh != null)
                {
                    string resolution_label = texture_width + "x" + texture_height;

                    Mesh boidMesh = GenerateBoidMesh(boidBaseMesh);
                    string boidMeshName = boidSetName + "-" + meshSubname + "-boids-mesh-" + resolution_label + ".mesh";
                    AssetDatabase.CreateAsset(boidMesh, outputPath + boidMeshName);
                    bool applyToPrefab = autoAssign && boidPrefab != null;
                    if (applyToPrefab)
                    {
                        ApplyMeshToPrefab(boidPrefab, boidMesh);
                    }
                }
            }
            if (GUILayout.Button("Apply Sim Shader"))
            {
                baseShaderSim = (Shader)AssetDatabase.LoadAssetAtPath(defaultSourceSimulatorShader, typeof(Shader));
                if (baseShaderSim != null)
                {
                    string resolution_label = texture_width + "x" + texture_height;

                    string simShaderFilePath = outputPath + boidSetName + "-" + simShaderSubname + "-sim-shader-" + resolution_label + ".shader";
                    string shaderName = boidSetName + "-" + simShaderSubname;
                    newSimShader = GenerateSimulatorShaderAsset(baseShaderSim, simShaderFilePath, shaderName, texture_width, texture_height);

                    bool useBaseMaterial = copySettingsFromMaterial && baseSimulatorMaterial != null;
                    Material generatedSimulatorMaterial = useBaseMaterial ? GenerateMaterial(newSimShader, baseSimulatorMaterial) : GenerateMaterial(newSimShader);

                    string simMaterialName = boidSetName + "-" + simShaderSubname + "-sim-mat-" + resolution_label + ".mat";
                    string simMaterialFilePath = outputPath + simMaterialName;
                    AssetDatabase.CreateAsset(generatedSimulatorMaterial, simMaterialFilePath);

                    bool applyToPrefab = autoAssign && boidPrefab != null;
                    if (applyToPrefab)
                    {
                        ApplySimulatorToPrefab(boidPrefab, generatedSimulatorMaterial);
                    }
                }
                else
                {
                    Debug.LogError("Failed to load template simulator shader. Please ensure it has not been moved or renamed.");
                    Debug.LogError("Expected file path: " + defaultSourceSimulatorShader);
                }
            }
            if (GUILayout.Button("Apply Boid Shader"))
            {
                baseShaderBoid = (Shader)AssetDatabase.LoadAssetAtPath(defaultSourceMeshboidShader, typeof(Shader));
                if (baseShaderBoid != null)
                {
                    string resolution_label = texture_width + "x" + texture_height;

                    string boidShaderName = boidSetName + "-" + boidShaderSubname + "-boid-shader-" + resolution_label + ".shader";
                    string boidShaderFilePath = outputPath + boidShaderName;
                    newBoidShader = GenerateBoidShaderAsset(baseShaderBoid, boidShaderFilePath);

                    bool useBaseMaterial = copySettingsFromMaterial && baseBoidMaterial != null;
                    Material generatedBoidMaterial = useBaseMaterial ? GenerateMaterial(newBoidShader, baseBoidMaterial) : GenerateMaterial(newBoidShader);

                    string simMaterialName = boidSetName + "-" + boidShaderSubname + "-boid-mat-" + resolution_label + ".mat";
                    string simMaterialFilePath = outputPath + simMaterialName;
                    AssetDatabase.CreateAsset(generatedBoidMaterial, simMaterialFilePath);

                    bool applyToPrefab = autoAssign && boidPrefab != null;
                    if (applyToPrefab)
                    {
                        ApplyBoidMaterialToPrefab(boidPrefab, generatedBoidMaterial);
                    }
                }
                else
                {
                    Debug.LogError("Failed to load template boid shader. Please ensure it has not been moved or renamed.");
                    Debug.LogError("Expected file path: " + defaultSourceMeshboidShader);
                }
            }
            if (GUILayout.Button("Generate RenderTexture"))
            {
                string resolution_label = texture_width + "x" + texture_height;


                RenderTexture renderTexture = new RenderTexture(texture_width * 2, texture_height * 4, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
                renderTexture.filterMode = FilterMode.Point;
                renderTexture.anisoLevel = 0;
                AssetDatabase.CreateAsset(renderTexture, outputPath + boidSetName + "-" + renderTextureSubname + "-rtex-" + resolution_label + ".renderTexture");

                bool applyToPrefab = autoAssign && boidPrefab != null;
                if (applyToPrefab)
                {
                    ApplyRenderTextureToPrefab(boidPrefab, renderTexture);
                }
            }




        }
    }
}