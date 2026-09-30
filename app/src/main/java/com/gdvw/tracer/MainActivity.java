package com.gdvw.tracer;

import android.app.Activity;
import android.os.Bundle;
import android.content.Intent;
import android.net.Uri;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.drawable.BitmapDrawable;
import android.graphics.drawable.Drawable;
import android.view.View;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.*;

import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.json.JSONObject;

import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int PICK_PNG = 1001;
    private static final int SAVE_GMD = 1002;
    private static final int SAVE_PREVIEW = 1003;

    private File inputFile;
    private File outputGmd;
    private File outputPreview;
    private ImageView preview;
    private TextView status;
    private Spinner profile;
    private Button traceButton;
    private Button saveGmdButton;
    private Button savePreviewButton;
    private final ExecutorService executor = Executors.newSingleThreadExecutor();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        if (!Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        setContentView(buildUi());
        showPreviousCrashStage();
    }

    private View buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(20), dp(18), dp(20), dp(28));
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("GD VW Tracer v5.18");
        title.setTextSize(26);
        title.setTextColor(Color.BLACK);
        title.setPadding(0, 0, 0, dp(4));
        root.addView(title);

        TextView subtitle = new TextView(this);
        subtitle.setText("Local PNG → Geometry Dash .gmd tracer • Android v0.8");
        subtitle.setTextSize(14);
        subtitle.setTextColor(Color.DKGRAY);
        subtitle.setPadding(0, 0, 0, dp(18));
        root.addView(subtitle);

        Button choose = new Button(this);
        choose.setText("Choose PNG");
        choose.setOnClickListener(v -> choosePng());
        root.addView(choose, matchWrap());

        TextView profileLabel = new TextView(this);
        profileLabel.setText("Profile");
        profileLabel.setTextSize(16);
        profileLabel.setTextColor(Color.BLACK);
        profileLabel.setPadding(0, dp(16), 0, dp(6));
        root.addView(profileLabel);

        profile = new Spinner(this);
        String[] profiles = {"Balanced", "Aggressive", "Quality"};
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_spinner_dropdown_item, profiles);
        profile.setAdapter(adapter);
        root.addView(profile, matchWrap());

        traceButton = new Button(this);
        traceButton.setText("Trace Locally");
        traceButton.setEnabled(false);
        traceButton.setOnClickListener(v -> trace());
        LinearLayout.LayoutParams traceLp = matchWrap();
        traceLp.topMargin = dp(14);
        root.addView(traceButton, traceLp);

        status = new TextView(this);
        status.setText("Select a PNG to begin.");
        status.setTextSize(14);
        status.setTextColor(Color.DKGRAY);
        status.setPadding(0, dp(12), 0, dp(12));
        root.addView(status);

        preview = new ImageView(this);
        preview.setAdjustViewBounds(true);
        preview.setScaleType(ImageView.ScaleType.FIT_CENTER);
        preview.setBackgroundColor(Color.rgb(240, 240, 240));
        LinearLayout.LayoutParams imageLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(360));
        root.addView(preview, imageLp);

        saveGmdButton = new Button(this);
        saveGmdButton.setText("Save .GMD");
        saveGmdButton.setEnabled(false);
        saveGmdButton.setOnClickListener(v -> saveFile(outputGmd, "trace_v5_18.gmd", "application/octet-stream", SAVE_GMD));
        LinearLayout.LayoutParams saveLp = matchWrap();
        saveLp.topMargin = dp(14);
        root.addView(saveGmdButton, saveLp);

        savePreviewButton = new Button(this);
        savePreviewButton.setText("Save Preview PNG");
        savePreviewButton.setEnabled(false);
        savePreviewButton.setOnClickListener(v -> saveFile(outputPreview, "trace_v5_18_preview.png", "image/png", SAVE_PREVIEW));
        root.addView(savePreviewButton, matchWrap());

        TextView privacy = new TextView(this);
        privacy.setText("Everything is processed on this device. No server upload is used.");
        privacy.setTextSize(12);
        privacy.setTextColor(Color.GRAY);
        privacy.setGravity(Gravity.CENTER_HORIZONTAL);
        privacy.setPadding(0, dp(20), 0, 0);
        root.addView(privacy);

        return scroll;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
    }

    private int dp(int n) {
        return Math.round(n * getResources().getDisplayMetrics().density);
    }

    private void choosePng() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("image/png");
        startActivityForResult(intent, PICK_PNG);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            if (requestCode == PICK_PNG) {
                inputFile = new File(getCacheDir(), "input.png");
                copyUriToFile(uri, inputFile);
                setPreviewFile(inputFile, 1024);
                status.setText("PNG loaded. Choose a profile and tap Trace Locally.");
                traceButton.setEnabled(true);
                saveGmdButton.setEnabled(false);
                savePreviewButton.setEnabled(false);
            } else if (requestCode == SAVE_GMD && outputGmd != null) {
                copyFileToUri(outputGmd, uri);
                status.setText("GMD saved.");
            } else if (requestCode == SAVE_PREVIEW && outputPreview != null) {
                copyFileToUri(outputPreview, uri);
                status.setText("Preview saved.");
            }
        } catch (Exception e) {
            status.setText("File error: " + e.getMessage());
        }
    }

    private void trace() {
        if (inputFile == null) return;
        traceButton.setEnabled(false);
        saveGmdButton.setEnabled(false);
        savePreviewButton.setEnabled(false);
        status.setText("Tracing locally… this can take a while for detailed art.");
        String selected = profile.getSelectedItem().toString().toLowerCase();
        File work = new File(getFilesDir(), "trace_work");

        clearPreviewBitmap();
        System.gc();

        executor.submit(() -> {
            try {
                Python py = Python.getInstance();
                PyObject module = py.getModule("mobile_bridge");
                PyObject result = module.callAttr("trace", inputFile.getAbsolutePath(), selected, work.getAbsolutePath());
                JSONObject obj = new JSONObject(result.toString());
                outputGmd = new File(obj.getString("gmd"));
                outputPreview = new File(obj.getString("preview"));
                int objects = obj.getInt("objects");
                double reduction = obj.optDouble("reduction_percent", 0.0);
                runOnUiThread(() -> {
                    setPreviewFile(outputPreview, 1024);
                    status.setText("Finished: " + objects + " objects • " + selected + " • " + String.format("%.1f%% reduction", reduction));
                    traceButton.setEnabled(true);
                    saveGmdButton.setEnabled(true);
                    savePreviewButton.setEnabled(true);
                });
            } catch (Exception e) {
                runOnUiThread(() -> {
                    status.setText("Trace failed: " + e.getMessage());
                    traceButton.setEnabled(true);
                });
            }
        });
    }

    private void clearPreviewBitmap() {
        Drawable old = preview.getDrawable();
        preview.setImageDrawable(null);
        if (old instanceof BitmapDrawable) {
            Bitmap oldBitmap = ((BitmapDrawable) old).getBitmap();
            if (oldBitmap != null && !oldBitmap.isRecycled()) {
                oldBitmap.recycle();
            }
        }
    }

    private void setPreviewFile(File file, int maxDimension) {
        if (file == null || !file.exists()) return;
        BitmapFactory.Options bounds = new BitmapFactory.Options();
        bounds.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(file.getAbsolutePath(), bounds);

        int sample = 1;
        int largest = Math.max(bounds.outWidth, bounds.outHeight);
        while (largest / sample > maxDimension * 2) {
            sample *= 2;
        }

        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inSampleSize = Math.max(1, sample);
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap = BitmapFactory.decodeFile(file.getAbsolutePath(), options);
        if (bitmap == null) return;

        Drawable old = preview.getDrawable();
        preview.setImageBitmap(bitmap);
        if (old instanceof BitmapDrawable) {
            Bitmap oldBitmap = ((BitmapDrawable) old).getBitmap();
            if (oldBitmap != null && oldBitmap != bitmap && !oldBitmap.isRecycled()) {
                oldBitmap.recycle();
            }
        }
    }

    private void showPreviousCrashStage() {
        File stage = new File(new File(getFilesDir(), "trace_work"), "crash_stage.txt");
        if (!stage.exists()) return;
        try (BufferedReader reader = new BufferedReader(new FileReader(stage))) {
            String line = reader.readLine();
            if (line != null && !line.startsWith("COMPLETE")) {
                status.setText("Previous trace stopped during: " + line);
            }
        } catch (IOException ignored) {
        }
    }

    private void saveFile(File source, String name, String mime, int requestCode) {
        if (source == null || !source.exists()) return;
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType(mime);
        intent.putExtra(Intent.EXTRA_TITLE, name);
        startActivityForResult(intent, requestCode);
    }

    private void copyUriToFile(Uri uri, File dest) throws IOException {
        try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(dest)) {
            if (in == null) throw new IOException("Could not open selected file");
            byte[] buf = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
    }

    private void copyFileToUri(File src, Uri uri) throws IOException {
        try (InputStream in = new FileInputStream(src); OutputStream out = getContentResolver().openOutputStream(uri, "w")) {
            if (out == null) throw new IOException("Could not open destination");
            byte[] buf = new byte[1024 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
    }
}
