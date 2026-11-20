import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const MainApp());
}

class MainApp extends StatefulWidget {
  const MainApp({super.key});

  @override
  State<MainApp> createState() => _MainAppState();
}

class _MainAppState extends State<MainApp> {
  Interpreter? _interpreter;
  Future<void>? _modelLoadFuture;

  File? _image;

  List<Detection> detections = [];

  List<String> labels = [];
  String _statusMessage = 'Memuat model...';

  // ============================================================
  // MODEL CONFIGURATION
  // ============================================================

  static const int inputSize = 640;

  // Sesuai graph:
  // 3 class + 4 bbox = 7
  static const int numClasses = 3;

  // Jumlah candidate:
  // 80x80 + 40x40 + 20x20
  // = 6400 + 1600 + 400
  // = 8400
  static const int numCandidates = 8400;

  static const double confThreshold = 0.40;
  static const double iouThreshold = 0.50;

  int originalWidth = 0;
  int originalHeight = 0;

  @override
  void initState() {
    super.initState();

    _modelLoadFuture = loadModel();
    loadLabels();
  }

  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  // ============================================================
  // LOAD MODEL
  // ============================================================

  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best.tflite');

      debugPrint("=================================");
      debugPrint("MODEL LOADED");
      debugPrint("Input : ${_interpreter!.getInputTensor(0).shape}");
      debugPrint("Output: ${_interpreter!.getOutputTensor(0).shape}");
      debugPrint("=================================");

      if (mounted) {
        setState(() {
          _statusMessage = 'Model siap. Pilih foto untuk mulai mendeteksi.';
        });
      }
    } catch (e) {
      debugPrint("ERROR LOAD MODEL: $e");
      if (mounted) {
        setState(() {
          _statusMessage = 'Gagal memuat model: $e';
        });
      }
    }
  }

  // ============================================================
  // LOAD LABELS
  // ============================================================

  Future<void> loadLabels() async {
    try {
      final data = await DefaultAssetBundle.of(
        context,
      ).loadString("assets/labels.txt");

      labels = data
          .split('\n')
          .map((e) => e.trim())
          .where((e) => e.isNotEmpty)
          .toList();

      debugPrint("Labels: $labels");
    } catch (e) {
      debugPrint("ERROR LOAD LABELS: $e");
    }
  }

  // ============================================================
  // PICK IMAGE
  // ============================================================

  Future<void> pickImage(ImageSource source) async {
    try {
      await _modelLoadFuture;

      if (_interpreter == null) {
        return;
      }

      final picker = ImagePicker();
      final XFile? picked = await picker.pickImage(source: source);

      if (picked == null) return;

      final file = File(picked.path);
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);

      if (decoded == null) {
        throw Exception('Gambar tidak dapat dibaca.');
      }

      originalWidth = decoded.width;
      originalHeight = decoded.height;

      setState(() {
        _image = file;
        detections = [];
        _statusMessage = 'Memproses gambar...';
      });

      await runModel(file);
    } catch (e) {
      debugPrint('ERROR PILIH/PROSES GAMBAR: $e');
      if (mounted) {
        setState(() {
          _statusMessage = 'Gagal memproses gambar: $e';
        });
      }
    }
  }

  // ============================================================
  // PREPROCESS
  // ============================================================

  Float32List preprocess(File imageFile) {
    final bytes = imageFile.readAsBytesSync();

    final image = img.decodeImage(bytes);

    if (image == null) {
      throw Exception("Tidak dapat membaca gambar");
    }

    // ==========================================================
    // CATATAN:
    // Ini menggunakan resize langsung 640x640.
    //
    // Jika model Anda dilatih/export menggunakan letterbox,
    // preprocessing harus diganti ke letterbox.
    // ==========================================================

    final resized = img.copyResize(image, width: inputSize, height: inputSize);

    final input = Float32List(1 * inputSize * inputSize * 3);

    int index = 0;

    for (int y = 0; y < inputSize; y++) {
      for (int x = 0; x < inputSize; x++) {
        final pixel = resized.getPixel(x, y);

        input[index++] = pixel.r / 255.0;
        input[index++] = pixel.g / 255.0;
        input[index++] = pixel.b / 255.0;
      }
    }

    return input;
  }

  // ============================================================
  // RUN MODEL
  // ============================================================

  Future<void> runModel(File imageFile) async {
    if (_interpreter == null) {
      throw StateError('Interpreter model belum siap.');
    }

    final inputShape = _interpreter!.getInputTensor(0).shape;
    final outputShape = _interpreter!.getOutputTensor(0).shape;
    const expectedInputShape = [1, inputSize, inputSize, 3];
    const expectedOutputShape = [1, 7, numCandidates];

    if (!_sameShape(inputShape, expectedInputShape)) {
      throw StateError(
        'Bentuk input model $inputShape tidak cocok dengan '
        '$expectedInputShape.',
      );
    }
    if (!_sameShape(outputShape, expectedOutputShape)) {
      throw StateError(
        'Bentuk output model $outputShape tidak cocok dengan '
        '$expectedOutputShape. Decoder saat ini hanya mendukung '
        '[1, 7, $numCandidates].',
      );
    }

    final input = preprocess(imageFile);

    // ==========================================================
    // OUTPUT SESUAI GRAPH:
    //
    // [1, 7, 8400]
    //
    // 0 = x
    // 1 = y
    // 2 = width
    // 3 = height
    //
    // 4 = class 0
    // 5 = class 1
    // 6 = class 2
    // ==========================================================

    final output = List.generate(
      1,
      (_) => List.generate(7, (_) => List<double>.filled(numCandidates, 0.0)),
    );

    // ==========================================================
    // INFERENCE
    // ==========================================================

    _interpreter!.run(input.reshape(inputShape), output);

    debugPrint(
      "Output shape = [${output.length}, "
      "${output[0].length}, "
      "${output[0][0].length}]",
    );

    // ==========================================================
    // DECODE
    // ==========================================================

    final List<Detection> results = [];
    double highestScore = 0;
    int highestClassIndex = -1;

    for (int i = 0; i < numCandidates; i++) {
      // --------------------------------------------------------
      // 1. BOUNDING BOX
      // --------------------------------------------------------

      double x = output[0][0][i];
      double y = output[0][1][i];
      double w = output[0][2][i];
      double h = output[0][3][i];

      // --------------------------------------------------------
      // 2. CLASS SCORE
      //
      // Graph sudah memiliki Logistic/Sigmoid.
      // Jadi JANGAN sigmoid lagi di sini.
      // --------------------------------------------------------

      double maxScore = 0.0;
      int classIndex = -1;

      for (int c = 0; c < numClasses; c++) {
        final score = output[0][4 + c][i];

        if (score > highestScore) {
          highestScore = score;
          highestClassIndex = c;
        }

        if (score > maxScore) {
          maxScore = score;
          classIndex = c;
        }
      }

      // --------------------------------------------------------
      // 3. CONFIDENCE FILTER
      // --------------------------------------------------------

      if (maxScore < confThreshold) {
        continue;
      }

      if (classIndex < 0) {
        continue;
      }

      // ========================================================
      // 4. KOORDINAT
      // ========================================================
      //
      // Berdasarkan graph, bbox branch sudah melakukan:
      //
      // center = (x1 + x2) * 0.5
      // size   = x2 - x1
      //
      // sehingga output akhir adalah XYWH.
      //
      // TFLite YOLO export umumnya memberikan koordinat
      // relatif terhadap input model atau koordinat input model,
      // sehingga kita cek skalanya.
      // ========================================================

      // Jika output berada pada range 0..1:
      if (x.abs() <= 1.5 &&
          y.abs() <= 1.5 &&
          w.abs() <= 1.5 &&
          h.abs() <= 1.5) {
        x *= originalWidth;
        y *= originalHeight;
        w *= originalWidth;
        h *= originalHeight;
      } else {
        // Jika sudah berupa pixel 640x640,
        // scale langsung ke ukuran gambar asli.
        x *= originalWidth / inputSize;
        y *= originalHeight / inputSize;
        w *= originalWidth / inputSize;
        h *= originalHeight / inputSize;
      }

      // --------------------------------------------------------
      // 5. CLAMP
      // --------------------------------------------------------

      w = max(0, w);
      h = max(0, h);

      if (w <= 1 || h <= 1) {
        continue;
      }

      results.add(
        Detection(
          x: x,
          y: y,
          w: w,
          h: h,
          confidence: maxScore,
          classIndex: classIndex,
        ),
      );
    }

    // ==========================================================
    // NMS
    // ==========================================================

    final finalDetections = nonMaxSuppression(results);

    debugPrint("Raw detections : ${results.length}");

    debugPrint("Final detections : ${finalDetections.length}");
    debugPrint(
      'Highest confidence: $highestScore '
      '(class $highestClassIndex), threshold: $confThreshold',
    );

    if (!mounted) return;

    setState(() {
      detections = finalDetections;
      if (finalDetections.isEmpty) {
        final scoreText = highestScore.toStringAsFixed(3);
        _statusMessage = results.isEmpty
            ? 'Tidak ada deteksi di atas batas confidence '
                  '$confThreshold. Tertinggi: $scoreText '
                  '(kelas $highestClassIndex).'
            : 'Ada kandidat, tetapi kotak deteksi tidak valid. '
                  'Periksa format koordinat output model.';
      } else {
        _statusMessage =
            'Terdeteksi ${finalDetections.length} objek. '
            'Confidence tertinggi: '
            '${finalDetections.first.confidence.toStringAsFixed(3)}.';
      }
    });
  }

  bool _sameShape(List<int> actual, List<int> expected) {
    if (actual.length != expected.length) return false;
    for (var i = 0; i < actual.length; i++) {
      if (actual[i] != expected[i]) return false;
    }
    return true;
  }

  // ============================================================
  // NMS
  // ============================================================

  List<Detection> nonMaxSuppression(List<Detection> boxes) {
    if (boxes.isEmpty) {
      return [];
    }

    // Urutkan confidence terbesar
    boxes.sort((a, b) => b.confidence.compareTo(a.confidence));

    final List<Detection> selected = [];

    for (final box in boxes) {
      bool keep = true;

      for (final existing in selected) {
        // Sebaiknya NMS per-class.
        if (box.classIndex != existing.classIndex) {
          continue;
        }

        if (iou(box, existing) > iouThreshold) {
          keep = false;
          break;
        }
      }

      if (keep) {
        selected.add(box);
      }
    }

    return selected;
  }

  // ============================================================
  // IOU
  // ============================================================

  double iou(Detection a, Detection b) {
    final double x1 = max(a.left, b.left);

    final double y1 = max(a.top, b.top);

    final double x2 = min(a.right, b.right);

    final double y2 = min(a.bottom, b.bottom);

    final double intersectionWidth = max(0.0, x2 - x1);

    final double intersectionHeight = max(0.0, y2 - y1);

    final double intersection = intersectionWidth * intersectionHeight;

    final double union = a.area + b.area - intersection;

    if (union <= 0) {
      return 0.0;
    }

    return intersection / union;
  }

  // ============================================================
  // UI
  // ============================================================

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text("YOLOv11 Detection")),
        body: Column(
          children: [
            // --------------------------------------------------
            // BUTTON
            // --------------------------------------------------

            Padding(
              padding: const EdgeInsets.all(8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  ElevatedButton(
                    onPressed: () {
                      pickImage(ImageSource.gallery);
                    },
                    child: const Text("Gallery"),
                  ),

                  ElevatedButton(
                    onPressed: () {
                      pickImage(ImageSource.camera);
                    },
                    child: const Text("Camera"),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
              child: Text(_statusMessage, textAlign: TextAlign.center),
            ),

            // --------------------------------------------------
            // IMAGE + BOX
            // --------------------------------------------------
            Expanded(
              child: _image == null
                  ? const Center(child: Text("No Image"))
                  : LayoutBuilder(
                      builder: (context, constraints) {
                        final double widgetWidth = constraints.maxWidth;

                        final double widgetHeight = constraints.maxHeight;

                        final double imageAspect =
                            originalWidth / originalHeight;

                        final double widgetAspect = widgetWidth / widgetHeight;

                        double displayWidth;
                        double displayHeight;

                        double offsetX = 0;
                        double offsetY = 0;

                        if (imageAspect > widgetAspect) {
                          displayWidth = widgetWidth;

                          displayHeight = widgetWidth / imageAspect;

                          offsetY = (widgetHeight - displayHeight) / 2;
                        } else {
                          displayHeight = widgetHeight;

                          displayWidth = widgetHeight * imageAspect;

                          offsetX = (widgetWidth - displayWidth) / 2;
                        }

                        final double scaleX = displayWidth / originalWidth;

                        final double scaleY = displayHeight / originalHeight;

                        return Stack(
                          children: [
                            // IMAGE
                            Positioned(
                              left: 0,
                              top: 0,
                              width: widgetWidth,
                              height: widgetHeight,
                              child: Image.file(_image!, fit: BoxFit.contain),
                            ),

                            // BOXES
                            ...detections.map((d) {
                              final String label = d.classIndex < labels.length
                                  ? labels[d.classIndex]
                                  : "class_${d.classIndex}";

                              return Positioned(
                                left: d.left * scaleX + offsetX,
                                top: d.top * scaleY + offsetY,
                                width: d.w * scaleX,
                                height: d.h * scaleY,
                                child: Container(
                                  decoration: BoxDecoration(
                                    border: Border.all(
                                      color: Colors.red,
                                      width: 2,
                                    ),
                                  ),
                                  child: Align(
                                    alignment: Alignment.topLeft,
                                    child: Container(
                                      color: Colors.red,
                                      padding: const EdgeInsets.symmetric(
                                        horizontal: 4,
                                        vertical: 2,
                                      ),
                                      child: Text(
                                        "$label "
                                        "${(d.confidence * 100).toStringAsFixed(1)}%",
                                        style: const TextStyle(
                                          color: Colors.white,
                                          fontSize: 12,
                                          fontWeight: FontWeight.bold,
                                        ),
                                      ),
                                    ),
                                  ),
                                ),
                              );
                            }),
                          ],
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}

// ================================================================
// DETECTION
// ================================================================

class Detection {
  final double x;
  final double y;
  final double w;
  final double h;

  final double confidence;

  final int classIndex;

  Detection({
    required this.x,
    required this.y,
    required this.w,
    required this.h,
    required this.confidence,
    required this.classIndex,
  });

  // x,y = CENTER

  double get left => x - (w / 2);

  double get top => y - (h / 2);

  double get right => x + (w / 2);

  double get bottom => y + (h / 2);

  double get area => max(0, w) * max(0, h);
}
