[app]
# (str) Title of your application
title = MyKivyApp

# (str) Package name
package.name = mykivyapp

# (str) Package domain (needed for android/ios packaging)
package.domain = org.example

# (str) Source code where the main.py file is located
source.dir = .

# (list) Application requirements
requirements = python3, kivy, onnxruntime, numpy, transformers

# (str) Supported orientation (one of: landscape, sensorLandscape, portrait or all)
orientation = portrait

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 1

# (list) Permissions
android.permissions = INTERNET

# (int) Target Android API, should be as high as possible.
android.api = 31

# (str) Presplash background color (for example `#FFFFFF`)
presplash_color = #FFFFFF

# (str) Presplash background image (png)
# presplash_img = %(source.dir)s/assets/presplash.png

# (str) Icon of the application
#icon.filename = %(source.dir)s/assets/icon.png

# (str) Version of your application
version = 0.1

# (str) Android NDK version to use
android.ndk = 25b

# (str) Android SDK version to use
android.sdk = 31
# (str) Android entry point
android.entrypoint = org.kivy.android.PythonActivity

# (str) Full name including package name of the Java class that implements PythonActivity
android.activity_class_name = org.kivy.android.PythonActivity

# (str) Extra xml to add to AndroidManifest.xml (will be added as child of <manifest> tag)
# android.manifest_xml = 

# (str) Extra xml to add to AndroidManifest.xml (will be added as child of <application> tag)
# android.manifest_application_xml = 

# (str) Gradle version to use
android.gradle_dependencies = com.android.support:appcompat-v7:28.0.0
