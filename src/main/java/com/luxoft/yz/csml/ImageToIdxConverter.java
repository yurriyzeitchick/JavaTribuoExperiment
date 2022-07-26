package com.luxoft.yz.csml;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author YZaychyk
 * @since 1.0
 **/
public class ImageToIdxConverter
{
    public static void convertPngToIdx(String input, String output, String subfolder, String outputFileName) throws IOException {
        var inputPath = Paths.get("f:\\events\\luxoft\\java_universe\\data_science\\ml_tribuo\\dataset\\", subfolder);
        var outputPath = Paths.get("f:\\events\\luxoft\\java_universe\\data_science\\ml_tribuo\\idx");

        if (!Files.isDirectory(inputPath))
        {
            return;
        }

        var labeledImages = new HashMap<String, List<BufferedImage>>();

        Files.list(inputPath).forEach(dir -> {
            try {
                labeledImages.put(dir.getFileName().toString(), readDir(dir));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });

        long imageCount = labeledImages.values().stream().mapToLong(List::size).sum();
        var firstImage = labeledImages.values().iterator().next().get(0);

        class Counter {
            int count = 0;

            int inc() { return ++count; }
        }

        var counter = new Counter();

        var imgOutPutFile = Files.createFile(Path.of(outputPath.toString(), outputFileName + "_images.idx3-long"));
        var labeOutPutFile = Files.createFile(Path.of(outputPath.toString(), outputFileName + "_labels.idx3-long"));
        var imgDOS = new DataOutputStream(Files.newOutputStream(imgOutPutFile));
        var labelDOS = new DataOutputStream(Files.newOutputStream(labeOutPutFile));
        imgDOS.writeShort(0);
        imgDOS.writeByte(0x0C);
        imgDOS.writeByte(3);
        imgDOS.writeInt((int) imageCount);
        imgDOS.writeInt(firstImage.getHeight());
        imgDOS.writeInt(firstImage.getWidth());

        labelDOS.writeShort(0);
        labelDOS.writeByte(0x0C);
        labelDOS.writeByte(1);
        labelDOS.writeInt((int) imageCount);

        labeledImages.forEach((label, images) -> {
            counter.inc();
            images.forEach(image -> {
                try {
                    labelDOS.writeInt(counter.count);
                    for (int y = 0; y < image.getHeight(); y++)
                        for (int x = 0; x < image.getWidth(); x++) {
                            imgDOS.writeInt(image.getRGB(x, y));
                        }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
        });

        labelDOS.flush();
        labelDOS.close();

        imgDOS.flush();
        imgDOS.close();

    }

    public static List<BufferedImage> readDir(Path dir) throws IOException {
        if (!Files.isDirectory(dir))
        {
            return null;
        }
        var images = new ArrayList<BufferedImage>();
        Files.list(dir).forEach(f -> {
            try {
                images.add(ImageIO.read(f.toFile()));
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
        return images;
    }
}
