package com.luxoft.yz.csml;

import org.tribuo.MutableDataset;
import org.tribuo.classification.Label;
import org.tribuo.classification.LabelFactory;
import org.tribuo.classification.evaluation.LabelEvaluator;
import org.tribuo.datasource.IDXDataSource;
import org.tribuo.interop.tensorflow.*;
import org.tribuo.interop.tensorflow.example.CNNExamples;
import org.tribuo.interop.tensorflow.example.MLPExamples;
import org.tribuo.util.Util;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;

/**
 * @author YZaychyk
 * @since 1.0
 **/
public class Main
{
    private static final Properties config;

    static {
        config = new Properties();
        try {
            config.load(Main.class.getClassLoader().getResourceAsStream("config.properties"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void main(String[] args) {
        try {
            // MNIST dataset
            trainAndTest(Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.train.images")),
                    Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.train.labels")),
                    Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.test.images")),
                    Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.test.labels")),
                    true
                    );

            // Crossroad images dataset
            trainAndTest(Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.train.images")),
                    Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.train.labels")),
                    Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.test.images")),
                    Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.test.labels")),
                    false
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void trainAndTest(Path trainImagesPath, Path trainLabelsPath, Path testImagesPath, Path testLabelsPath, boolean tryCNN) throws IOException {
        var labelFactory = new LabelFactory();
        var labelEval = new LabelEvaluator();

        var imgTrainSource = new IDXDataSource<>(trainImagesPath, trainLabelsPath, labelFactory);

        var imgTestSource = new IDXDataSource<>(testImagesPath, testLabelsPath, labelFactory);


        var imgTrain = new MutableDataset<>(imgTrainSource);
        var imgTest = new MutableDataset<>(imgTestSource);

        var inputName = "MNIST_INPUT";
        var imgMLTuple = MLPExamples.buildMLPGraph(
                inputName,
                imgTrain.getFeatureMap().size(),
                new int[]{300, 200, 30},
                imgTrain.getOutputs().size()
        );

        var gradAlgorithm = GradientOptimiser.ADAGRAD;
        var gradParams = Map.of("learningRate", 0.01F, "initialAccumulatorValue", 0.01F);


        var imgDenseConverter = new DenseFeatureConverter(inputName);
        var imgOutputConverter = new LabelConverter();

        var imgMLPTrainer = new TensorFlowTrainer<Label>(imgMLTuple.graphDef,
                imgMLTuple.outputName,
                gradAlgorithm,
                gradParams,
                imgDenseConverter,
                imgOutputConverter,
                16,
                20,
                16,
                -1
        );

        var mlpStart = System.currentTimeMillis();
        var mlpModel = imgMLPTrainer.train(imgTrain);
        var mlpEnd = System.currentTimeMillis();
        System.out.println("Images MLP training took: " + Util.formatDuration(mlpStart, mlpEnd));

        if (tryCNN) {
            var cnnTuple = CNNExamples.buildLeNetGraph(inputName, 28, 255, imgTrain.getOutputs().size());
            var imageConverter = new ImageConverter(inputName, 28, 28, 1);

            var imgCNNTrainer = new TensorFlowTrainer<Label>(
                    cnnTuple.graphDef,
                    cnnTuple.outputName,
                    gradAlgorithm,
                    gradParams,
                    imageConverter,
                    imgOutputConverter,
                    16,
                    5,
                    16,
                    -1
            );

            var cnnStart = System.currentTimeMillis();
            var cnnModel = imgCNNTrainer.train(imgTrain);
            var cnnEnd = System.currentTimeMillis();
            System.out.println("CNN training took:" + Util.formatDuration(cnnStart, cnnEnd));
        }

        var mlpEvaluation = labelEval.evaluate(mlpModel, imgTest);
        System.out.println(mlpEvaluation);
        System.out.println(mlpEvaluation.getConfusionMatrix());
        System.out.println(mlpModel.getProvenance().toString());
    }


}
