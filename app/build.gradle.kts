plugins {
    id("com.android.application")
    id("com.chaquo.python")
}

android {
    namespace = "com.gdvw.tracer"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.gdvw.tracer"
        minSdk = 24
        targetSdk = 36
        versionCode = 4
        versionName = "0.4.0"
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }
}

chaquopy {
    defaultConfig {
        version = "3.10"
        buildPython("python3.10")
        pip {
            install("numpy==1.23.3")
            install("opencv-python-headless==4.5.1.48")
            install("Pillow==11.0.0")
            install("Shapely==1.8.5")
        }
        extractPackages("*")
    }
}
