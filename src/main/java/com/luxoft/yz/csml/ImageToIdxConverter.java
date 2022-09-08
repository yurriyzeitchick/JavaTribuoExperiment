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

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * @author YZaychyk
 * @since 1.0
 **/
public class ImageToIdxConverter
{
    public static void convertPngToIdx(Path inputPath, Path outputPath, String outputFileName, boolean isGrayScale) throws IOException {
        if (!Files.isDirectory(inputPath))
        {
            return;
        }

        var labeledImages = new HashMap<String, List<BufferedImage>>();

        try (var directories = Files.list(inputPath)) {
            directories.forEach(dir -> {
                try {
                    labeledImages.put(dir.getFileName().toString(), readDir(dir));
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            });
        }

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
        var idxDataTypeCode = (isGrayScale ? IdxDataType.UNSIGNED_BYTE : IdxDataType.INT).typeCode;

        imgDOS.writeShort(0);
        imgDOS.writeByte(idxDataTypeCode);
        imgDOS.writeByte(3);
        imgDOS.writeInt((int) imageCount);
        imgDOS.writeInt(firstImage.getHeight());
        imgDOS.writeInt(firstImage.getWidth());

        labelDOS.writeShort(0);
        labelDOS.writeByte(idxDataTypeCode);
        labelDOS.writeByte(1);
        labelDOS.writeInt((int) imageCount);

        labeledImages.forEach((label, images) -> {
            counter.inc();
            images.forEach(image -> {
                try {
                    labelDOS.writeByte(counter.count);
                    for (int y = 0; y < image.getHeight(); y++)
                        for (int x = 0; x < image.getWidth(); x++) {
                            writeImageData(image.getRGB(x,y), imgDOS, isGrayScale);
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

    private static void writeImageData(int pixelData, DataOutputStream dos, boolean isGrayScale) throws IOException {
        if (isGrayScale) {
            dos.writeByte(pixelData & 0xFF);
        } else {
            dos.writeInt(pixelData);
        }
    }

    private enum IdxDataType
    {
        UNSIGNED_BYTE(0x08),
        INT(0x0C);

        private final byte typeCode;

        IdxDataType(int typeCode) {
            this.typeCode = (byte)typeCode;
        }
    }
}
