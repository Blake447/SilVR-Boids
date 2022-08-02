using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using System;
using System.IO;

namespace SilVR
{

    class BoidAssignAssets : EditorWindow
    {
        GameObject boidPrefab = null;

        Material simMaterial = null;
        Material boidMaterial = null;
        Mesh boidMesh = null;
        RenderTexture simulatorTexture = null;

        [MenuItem("SilVR/Boids/Assign Assets")]

        public static void ShowWindow()
        {
            EditorWindow.GetWindow(typeof(BoidAssignAssets));
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

        private void OnGUI()
        {

            //autoAssign = EditorGUILayout.Toggle("Auto Assign Assets", autoAssign);

            GUILayout.Label("Prefab to Apply Changes To");
            boidPrefab = (GameObject)EditorGUILayout.ObjectField("Existing boid prefab", boidPrefab, typeof(GameObject), true);


            GUILayout.Space(8.0f);
            GUILayout.Label("Boid Asset Set", EditorStyles.boldLabel);

            //Material simMaterial = null;
            //Material boidMaterial = null;
            //Mesh boidMesh = null;
            //RenderTexture simulatorTexture = null;

            boidMesh = (Mesh)EditorGUILayout.ObjectField("Boid base mesh", boidMesh, typeof(Mesh), true);
            boidMaterial = (Material)EditorGUILayout.ObjectField("Base Boid Material", boidMaterial, typeof(Material), true);
            simMaterial = (Material)EditorGUILayout.ObjectField("Base Simulator Material", simMaterial, typeof(Material), true);
            simulatorTexture = (RenderTexture)EditorGUILayout.ObjectField("Base Simulator Material", simulatorTexture, typeof(RenderTexture), true);

            GUILayout.Space(4.0f);
            GUILayout.Label("Assign Assets", EditorStyles.boldLabel);

            if (GUILayout.Button("Generate All Assets"))
            {
                if (boidPrefab != null)
                {
                    if (boidMesh != null)
                    {
                        ApplyMeshToPrefab(boidPrefab, boidMesh);
                    }
                    if (boidMaterial != null)
                    {
                        ApplyBoidMaterialToPrefab(boidPrefab, boidMaterial);
                    }
                    if (simMaterial != null)
                    {
                        ApplySimulatorToPrefab(boidPrefab, simMaterial);
                    }
                    if (simulatorTexture != null)
                    {
                        ApplyRenderTextureToPrefab(boidPrefab, simulatorTexture);
                    }
                }
                else
                {
                    Debug.LogError("No boid prefab assigned");
                }
            }
        }
    }
}