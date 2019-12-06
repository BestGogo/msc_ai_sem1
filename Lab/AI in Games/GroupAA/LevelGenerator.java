package levelGenerators.GroupAA;

import engine.core.MarioLevelModel;
import engine.helper.MarioTimer;
import levelGenerators.MarioLevelGenerator;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;

public class LevelGenerator implements MarioLevelGenerator {

    private static final int rows = 16;
    private static final int columns = 150;
    private static int NTD =  2; // number of files to be read for training
    private static int type = 1; //map type (notch, linear or benWeber)

    public LevelGenerator(int type, int NTD){
        this.NTD = NTD;
        this.type = type;
    }

    public static char[][][] generateTraindata(int type, int kitna) throws IOException {
        String path = new String();
        String read_data = new String();
        char[][][] traindata = new char[rows][columns][kitna*2];
        int r = 0;
        int c = 0;
        int num = 0;
        for (int n=1; n<=kitna;n++) {
            // type 1 = notch

            if (type == 1) {
                path = "src\\levelGenerators\\GroupAA\\TrainLevels\\notch\\level";
            }

            // type 2 = linear

            if (type == 2) {
                path = "src\\levelGenerators\\GroupAA\\TrainLevels\\linear\\level";
            }

            // type 3 = benWeber

            if (type == 3) {
                path = "src\\levelGenerators\\GroupAA\\TrainLevels\\benWeber\\level";
            }
            path = path + n + ".txt";
            FileInputStream padh_re = new FileInputStream(path);
            Scanner gmd = new Scanner(padh_re);
            r=0;
            while (gmd.hasNext()) {
                if (r >= rows) {
//                    System.out.println("overboard");
                    r = 0;
                    num++;
                }
                read_data = gmd.next();
//                System.out.println(read_data);
                for (int y = 0; y < read_data.length(); y++) {
//                    System.out.println("in loop");
                    traindata[r][c][num] = read_data.charAt(y);
//                    System.out.println(traindata[num][r][c]);
                    c++;
                }

                c = 0;
                r++;
            }
            gmd.close();
            padh_re.close();
//            System.out.println("file_read");
            num++;
        }
        System.out.println(num);
        return traindata;
    }
//    public void markovchain (int x, int y)
    public static void main(String[] args) throws IOException {
       char[][][] traindata = generateTraindata(type,NTD);
       int num_train_levels = NTD*2;
       for(int a=0;a<num_train_levels;a++) {
           System.out.println("file" + (a+1));
           for(int b=0;b<rows;b++) {
               for(int c=0;c<columns;c++) {
                   System.out.print(traindata[b][c][a]);

               }
               System.out.println();
           }
       }
    }
    @Override
    public String getGeneratedLevel(MarioLevelModel model, MarioTimer timer) {
        return null;
    }

    @Override
    public String getGeneratorName() {
        return "GroupAA Level Generator";
    }
}
