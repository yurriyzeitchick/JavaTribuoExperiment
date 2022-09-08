/* This file is a part of csml.
 * csml is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
 * of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
 */
package com.luxoft.yz.csml;

import org.tribuo.Model;
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
import java.nio.file.Paths;
import java.util.Map;
import java.util.Properties;
import java.util.stream.Stream;

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
        if (hasParam("-c", args)) {
            convertPngToIdx();
        } else if (hasParam("-e", args)) {
            experiment();
        } else {
            printHelp();
        }
    }

    private static void convertPngToIdx() {
        try {
            var imagesDir = config.getProperty("png.grayscale.images.path");
            var idxOutPutPath = Paths.get(config.getProperty("experiment.idx.folder"));
            ImageToIdxConverter.convertPngToIdx(Paths.get(imagesDir, "train"), idxOutPutPath, "train", true);
            ImageToIdxConverter.convertPngToIdx(Paths.get(imagesDir, "test"), idxOutPutPath, "test", true);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static void experiment() {
        try {
            // MNIST dataset
            /*trainAndTest(Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.train.images")),
                    Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.train.labels")),
                    Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.test.images")),
                    Path.of(config.getProperty("mnist.idx.folder"), config.getProperty("mnist.idx.test.labels")),
                    true
                    );*/

            // Crossroad images dataset
            trainAndTest(Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.train.images")),
                    Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.train.labels")),
                    Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.test.images")),
                    Path.of(config.getProperty("experiment.idx.folder"), config.getProperty("experiment.idx.test.labels")),
                    false, true
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static void trainAndTest(Path trainImagesPath, Path trainLabelsPath, Path testImagesPath, Path testLabelsPath,
                                    boolean tryMlp, boolean tryCNN) throws IOException
    {
        var labelFactory = new LabelFactory();

        var imgTrainSource = new IDXDataSource<>(trainImagesPath, trainLabelsPath, labelFactory);
        var imgTestSource = new IDXDataSource<>(testImagesPath, testLabelsPath, labelFactory);

        var imgTrain = new MutableDataset<>(imgTrainSource);
        var imgTest = new MutableDataset<>(imgTestSource);

        var inputName = "MNIST_INPUT";

        var gradAlgorithm = GradientOptimiser.ADAGRAD;
        var gradParams = Map.of("learningRate", 0.01F, "initialAccumulatorValue", 0.01F);

        if (tryMlp)
            doMlpTest(inputName, imgTrain, imgTest, gradAlgorithm, gradParams);

        if (tryCNN)
            doCnnTest(inputName, imgTrain, imgTest, gradAlgorithm, gradParams);

    }

    private static void doMlpTest(String inputName, MutableDataset<Label> trainSet, MutableDataset<Label> testSet,
                           GradientOptimiser gradAlgorithm, Map<String, Float> gradParams)
    {
        var imgMLTuple = MLPExamples.buildMLPGraph(
                inputName,
                trainSet.getFeatureMap().size(),
                new int[]{300, 200, 30},
                trainSet.getOutputs().size()
        );

        var imgDenseConverter = new DenseFeatureConverter(inputName);
        var imgOutputConverter = new LabelConverter();

        var imgMLPTrainer = new TensorFlowTrainer<Label>(imgMLTuple.graphDef,
                imgMLTuple.outputName,
                gradAlgorithm,
                gradParams,
                imgDenseConverter,
                imgOutputConverter,
                16,
                100,
                16,
                -1
        );

        var mlpStart = System.currentTimeMillis();
        var mlpModel = imgMLPTrainer.train(trainSet);
        var mlpEnd = System.currentTimeMillis();

        System.out.println("Images MLP training took: " + Util.formatDuration(mlpStart, mlpEnd));

        evaluate(mlpModel, testSet);
    }

    private static void doCnnTest (String inputName, MutableDataset<Label> trainSet, MutableDataset<Label> testSet,
                            GradientOptimiser gradAlgorithm, Map<String, Float> gradParams)
    {
        var cnnTuple = CNNExamples.buildLeNetGraph(inputName, 400, 255, trainSet.getOutputs().size());
        var imageConverter = new ImageConverter(inputName, 400, 400, 1);

        var imgCNNTrainer = new TensorFlowTrainer<Label>(
                cnnTuple.graphDef,
                cnnTuple.outputName,
                gradAlgorithm,
                gradParams,
                imageConverter,
                new LabelConverter(),
                16,
                30,
                16,
                -1
        );

        var cnnStart = System.currentTimeMillis();
        var cnnModel = imgCNNTrainer.train(trainSet);
        var cnnEnd = System.currentTimeMillis();

        System.out.println("CNN training took:" + Util.formatDuration(cnnStart, cnnEnd));

        evaluate(cnnModel, testSet);
    }

    private static void evaluate(Model<Label> model, MutableDataset<Label> testSet) {
        var labelEval = new LabelEvaluator();
        var cnnEvaluation = labelEval.evaluate(model, testSet);

        System.out.println(cnnEvaluation);
        System.out.println(cnnEvaluation.getConfusionMatrix());
        System.out.println(model.getProvenance().toString());
    }

    private static boolean hasParam(String param, String[] args) {
        return Stream.of(args).anyMatch(arg -> arg.equalsIgnoreCase(param));
    }

    private static void printHelp() {
        var sb = new StringBuilder("Please use one of the command line parameters:\n");
        sb.append("[-c] convert PNG images to IDX format using paths defined in config.properties\n");
        sb.append("[-e] run the experiment using MNIST dataset and using crossroads dataset. Paths defined in config.properties\n");
        System.out.println(sb);
    }
}
