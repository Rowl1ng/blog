---
description: "VR Development"
title:  "Sketch VR Development Log"
layout: blog
tech: true
background-image: http://static.zybuluo.com/sixijinling/ndghatp2hshzuvinrvq96c9a/colorpicker.PNG
istop: true
date: 2020-1-22 14:00
update: 2020-1-22 14:00
category: tech
tags:
- VR
- Unity
---




# Raw Materials

- Devices:
    - Oculus Rift: headset + handset x 2 + sensor x 2 
    - Dell desktop with Nvidia 1070 GPU 
        - System： Windows 10
        - Development Software：Unity 2019.2.17f1 + Visual Studio 2019
- Language: C# 

Actually it is not only a sketch game. it can also be used for viewing 3D models (For example, VR online shopping for furnitures!). If you are inrerested, it can be further developed into a 3D prototyping tool after adding editing function.

# Data Preparation

In this game, players need to sketch the outline of the reference model. So our first task is to prepare our reference dataset. Here I list two most commonly used dataset:

- Modelnet40, which contains 12311 uniformly orientated `.off` format models from 40 categories, 
    - To make them compatible with Unity and easy to use, I normalize these models first and then convert them to `.obj` format. The normalization is composed of two steps:
1. translate to origin
2. scale to unit
Now we can assure that every model imported has the same origin and scale. Then I use `pyassimp` to save them as `.obj` file.

- ShapeNetCore v1: It contains 6778 `.obj` chair models.

# How to draw a line in VR?

The answer lies in the `PointLineManager.cs`. When pressing the trigger on your controller while moving it. The script will record the position of the controller and use it as new point of the line.

A basic implementation is to use `LineRenderer`. When player keep pressing the trigger we just keep adding new points to the current LineRenderer using the coordinate of contoller.

Another way is to use `MeshLineRenderer.cs`. Different form the former, it will create a new mesh which not only has vertices but also has triangle faces. So in this way you can save your sketch as obj while the `LineRenderer` which doesn't have meshes can only be saved as .off file.

# Basics of Unity

Unity has two modes: Editor & Build. In my practice, I found some functions can only run normally inside Editor mode. For example,  `AssetDatabase` is from `UnityEditor` namespace which is unavailable in build version. Here is a [Solution on StackOverflow][1].

## Scene

If you create an Unity 3D project from scratch, you will start from a defaulted scene where you can create any gameobject you want. In this VR project, we hope to collect some settings before players start to sketch, so we add a 2D start scene before the VR scene.

```

```

To select filepath and directory as saving path and model path, one choice is official FileDialog API `EditorUtility.OpenFilePanel` , but it can only be used in Editor mode. In build mode, we choose `StandaloneFileBrowser` Asset as substitute. You can search and import this asset in Unity asset store.

```
PlayerManager.save_dir = StandaloneFileBrowser.OpenFolderPanel("Select Save Folder", "", true)[0];

PlayerManager.namelist_path = StandaloneFileBrowser.OpenFilePanel("Open Namelist File", "", "txt", true)[0];

```

## GameObject

Unity has two "kinds" of gameobject: "Real" physical object & "virtual" script. If you want your script to function during the game, you can create an `Empty Object` and bind your script to it through `Add Component`.

If you want to operate on other object in current script. you can use `GameObject.Find("the name of other object")`

If you want to operate on other script in current script. you can use `MyObject.GetComponent<YourScriptName>()`



## File I/O

1. How to import .obj files into the game?

Search for [Runtime OBJ Importer][2] in Asset Store. Remenber to read the Readme file: You must go to the "Graphics" tab on the right side of Unity Editor, and add "Standard (Specular Setup)" to the `always included shaders`. This will make your next build take a while

```
targetPath = "absolute path"
loadedObject = new OBJLoader().Load(targetPath);
```

Another way is to use `Resources.Load()` to load .obj file in build mode. In this case, all model files should lie in `/Resources` directory （You can put`/Resources` directory under any subdirectory of your project）. But I found some models are displayed abnormally in this method.

```
targetpath = "path under Resources directory"
loadedObject = Instantiate(Resources.Load(targetPath) as GameObject);

```

2. How to save sketch?

I implemented two methods to save sketch: 

- export whole multiple lines as single mesh file, which has both vertices and triangle meshes.
- save sketch as trajectory：.off file with points coordinate only.

Meanwhile, we also save timestamps of strokes: a list of (x,y,z,timestamp)

In editor mode, it is defaulted to save 3 types of files: model+sketch, sketch, model. But in buld version, we only save sketch since meshes loaded by Resources is unallowed to be read or saved.


### WorldPosition vs. LocalPosition

Since you are not sure the position you are processing is local or World position, the safest way is to transform them into world position first and then do whatever you want.

For example, I need to save vertices of both sketch and model in one single obj file. So, to make sure they share the same coordinate system I first convert their position into world position `wv` and then to relative position `rv`.
```
Vector3 wv = mf.transform.TransformDirection(lv);
Vector3 rv = space.transform.InverseTransformDirection(wv);
```

# VR development (Oculus Rift)

Go to asset store and import `Oculus` asset.

Get input from controller:

```
float RI = OVRInput.Get(OVRInput.Axis1D.SecondaryIndexTrigger); //IndexTrigger on right hand controller
bool undoX = OVRInput.GetDown(OVRInput.RawButton.X); //X button on left controller
```

![此处输入图片的描述][3]

[More information about Input for Oculus][4]

## VR UIs

We can still use traditional `Canvas` inside VR. But to interact with canvas, mouse activity for 2D display should be replaced by `LaserPointer` for VR, which you can utilize easily by importing `UIHelper` prefab from Oculus Asset as a gameobject . 

Tutorial: [how to build vr uis with unity and oculus rift part 2][5]


## Grab/Rotate Objects

Add `OVR Grabbable` script to the object with `Rigid` and `Collider` component that you want to manipulate. And you can use your handtrigger to grab and rotate them.

For example, in this game I put both sketch and model under the same parent gameobject `space`, which means I only need to add `OVR Grabbable` script to `space` to manipulate both the sketch and model.

## OVRCameraRig

The `OVRCameraRig` prefab from Oculus Asset contains the camera for VR and two hand controller gameobjects. You can customize any controller operation with it easily. In my case, I attached a ColorPicker to left-hand controller.

# Other Functions

- ColorPicker: Modified from `SimpleColorPicker` Asset. Main modification is to transfer interaction from mouse drag to laserpoint drag, which is implemented in `Draggble.cs` under SimpleColorPicker/Script.
- CountDown: `CountDownScript`

# Tutorials

 - [Making VR game in Unity][6]
 - [Unity VR Tutorial: How To Build Tilt Brush From Scratch][7]
 - Basics of 3D graphics: Details are listed in [my wiki][8].

# Comments

- Oculus's development ecosystem is not friendly enough for beginners nowadays. Resources for steamVR seems better，at least it has more tutorials on youtube. But I finally returned to Oculus integration after having some input trouble with steamvr.


  [1]: https://stackoverflow.com/questions/40064746/the-type-or-namespace-name-unityeditor-could-not-be-found
  [2]: https://assetstore.unity.com/packages/tools/modeling/runtime-obj-importer-49547#reviews
  [3]: https://docs.unity3d.com/uploads/Main/OculusControllersTouchControllers.jpg
  [4]: https://docs.unity3d.com/Manual/OculusControllers.html
  [5]: https://arvrjourney.com/how-to-build-vr-uis-with-unity-and-oculus-rift-part-2-69e31b32dd82
  [6]: https://www.youtube.com/channel/UCPJlesN59MzHPPCp0Lg8sLw
  [7]: https://www.youtube.com/watch?v=eMJATZI0A7c
  [8]: https://rowl1ng.com/MyWiki/computer-graphics/3d-model.html

