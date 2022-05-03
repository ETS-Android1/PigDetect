/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.tsdetector.pigdetector.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import com.tsdetector.pigdetector.detection.customview.OverlayView;
import com.tsdetector.pigdetector.detection.env.BorderedText;
import com.tsdetector.pigdetector.detection.env.ImageUtils;
import com.tsdetector.pigdetector.detection.env.Logger;
import com.tsdetector.pigdetector.detection.ml.ModelPigdetector;
import com.tsdetector.pigdetector.detection.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.tensorflow.lite.examples.detection.tflite.Detector;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 320;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Detector detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;
  private Bitmap bitMapPig;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private ModelPigdetector model;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this, this.listView);

    int cropSize = TF_OD_API_INPUT_SIZE;


    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    bitMapPig = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new OverlayView.DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            try {
              model = ModelPigdetector.newInstance(getApplicationContext());
              // Creates inputs for reference.
              TensorImage normalizedInputImageTensor = TensorImage.fromBitmap(croppedBitmap);

              // Runs model inference and gets result.
              ModelPigdetector.Outputs outputs = model.process(normalizedInputImageTensor);
              TensorBuffer locations = outputs.getLocationsAsTensorBuffer();
              TensorBuffer classes = outputs.getClassesAsTensorBuffer();
              TensorBuffer scores = outputs.getScoresAsTensorBuffer();
              TensorBuffer numberOfDetections = outputs.getNumberOfDetectionsAsTensorBuffer();

              float[] locationsValues = locations.getFloatArray();
              float[] scoresValues = scores.getFloatArray();
              int sizeEachBox = locations.getShape()[locations.getShape().length - 1];

              Log.d("locations FlatSize---->" , String.valueOf(locations.getFlatSize()));
//              System.out.println("SCORESSSSSSSSSSS---->" + Arrays.toString(locations.getShape()));
              Log.d("locations shape ----->" , Arrays.toString(Arrays.copyOf(locations.getShape(), 5)));
              Log.d("scores length----->" , String.valueOf(scores.getFloatArray().length));
              Log.d("locations boxes----->" , Arrays.toString(Arrays.copyOf(locationsValues, 4)));
              Log.d("scores values ---->" , Arrays.toString(Arrays.copyOf(scores.getFloatArray(), 5)));


              LOGGER.i("Running detection on image " + currTimestamp);
              final long startTime = SystemClock.uptimeMillis();
              lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

              cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
              final Canvas canvas = new Canvas(cropCopyBitmap);
              final Paint paint = new Paint();
              paint.setColor(Color.RED);
              paint.setStyle(Style.STROKE);
              paint.setStrokeWidth(2.0f);

              float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
              switch (MODE) {
                case TF_OD_API:
                  minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                  break;
              }

              final List<Detector.Recognition> mappedRecognitions =
                      new ArrayList<Detector.Recognition>();

              for (int i = 0; i < scoresValues.length; i++) {
                int initPositionBoxCurrent = i * sizeEachBox;
                float[] box = Arrays.copyOfRange(locationsValues, initPositionBoxCurrent, initPositionBoxCurrent + sizeEachBox);
                System.out.println(scoresValues[i]);
                if (scoresValues[i] >= minimumConfidence) {
                  int left = (int) (box[1] * TF_OD_API_INPUT_SIZE);
                  int top = (int) (box[2] * TF_OD_API_INPUT_SIZE);
                  int right = (int) (box[3] * TF_OD_API_INPUT_SIZE);
                  int bottom = (int) (box[0] * TF_OD_API_INPUT_SIZE);
                  RectF rectF = new RectF(left, top, right, bottom);
                  canvas.drawRect(rectF, paint);
                  cropToFrameTransform.mapRect(rectF);
                  mappedRecognitions.add(new Detector.Recognition(i + "", " " + (i + 1), scoresValues[i], rectF));
                  System.out.println(Arrays.toString(box));
                }
              }

              tracker.trackResults(mappedRecognitions, currTimestamp);
              trackingOverlay.postInvalidate();

              Thread.sleep(7000);

              computingDetection = false;

              runOnUiThread(
                      new Runnable() {
                        @Override
                        public void run() {
                          showFrameInfo(previewWidth + "x" + previewHeight);
                          showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                          showInference(lastProcessingTimeMs + "ms");
                        }
                      });

              // Releases model resources if no longer used.
              model.close();
            } catch (IOException e) {
              e.printStackTrace();
            } catch (InterruptedException e) {
              e.printStackTrace();
            }
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(
        () -> {
          try {
            detector.setUseNNAPI(isChecked);
          } catch (UnsupportedOperationException e) {
            LOGGER.e(e, "Failed to set \"Use NNAPI\".");
            runOnUiThread(
                () -> {
                  Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
                });
          }
        });
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
