# Blockbuster OBJ (.bobj)

BOBJ is an extended version of the OBJ format that supports:

- Vertex weights  
- Armature (bone) data  
- Bone animations (keyframes)  

It is mainly used by the **[S&B](https://github.com/mchorse/snb)** And **[BBS Mod](https://github.com/mchorse/bbs-mod)** Minecraft mod.

---

## About

This add-on was originally created by **[McHorse](https://github.com/mchorse),** and based on Blender’s bundled `io_scene_obj` add-on (GPL licensed).

I did NOT create the original exporter.

What was added/changed:

- Added full **BOBJ import support**
- Added extra improvements and fixes
- Updated the add-on to properly work with **Blender 4.1**

Original creator: **[McHorse](https://github.com/mchorse),**  
Updated & extended for Blender 4.1: **[Eleeter](https://github.com/Eleeterlary)**

---
## New Features & Updates

- Full BOBJ Importer: You can now bring .bobj files back into Blender. It restores the 3D model, the bones, the vertex groups, and the animations.
- Blender 4.1+ Updated: the code to work with Blender 4.1's new way of handling "normals" so your models look right.
- Quaternion to Euler (XYZ): Added a checkbox to turn rotation data into simple XYZ math that Minecraft understands better. **You don't need to use the Rigify plugin anymore.**
- Auto Texture Export: The script can now automatically paste the texture to the folder where you save your model.

---

## Installation (Blender 4.1+)

Open Blender.

- Go to Edit > Preferences > Add-ons.

- Click Install...

- Select the downloaded file.

- Check the box for Blockbuster OBJ (.bobj).

---
## Once enabled, you will see the options under:

- **File > Import > Blockbuster OBJ (.bobj)**
- **File > Export > Blockbuster OBJ (.bobj)**

### If it does not appear, make sure the add-on is enabled in the Add-ons list.
---

License
- This add-on is based on Blender’s original code and is licensed under the GPL.

- The source code includes the license header.
---
