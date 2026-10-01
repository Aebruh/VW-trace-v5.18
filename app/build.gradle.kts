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
        versionCode = 10
        versionName = "0.10.0"
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
            // Custom arm64 Android wheels are created by the GitHub build job.
            options("--find-links", "${project.projectDir}/wheels")
            install("numpy==1.26.2")
            install("opencv-python-headless==4.5.1.48")
            install("Pillow==11.0.0")
            install("chaquopy-geos==3.9.4")
            install("Shapely==2.1.2")
        }
        extractPackages("*")
    }
}
