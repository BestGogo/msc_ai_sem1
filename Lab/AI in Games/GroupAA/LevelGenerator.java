package levelGenerators.GroupAA;

import engine.core.MarioLevelModel;
import engine.helper.MarioTimer;
import levelGenerators.MarioLevelGenerator;

import java.io.*;
import java.util.Arrays;
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

    public static char[][][] generateTrainData(int type, int kitna) throws IOException {
        String path = new String();
        String read_data = new String();
        char[][][] traindata = new char[rows][columns][kitna];
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

    public char[] dependencies(int x, int y, char[][] data){
        char[] depends = new char[3];
        depends[0] = data[x-1][y];
        depends[1] = data[x-1][y-1];
        depends[2] = data[x][y-1];
        return depends;
    }
    public double[] markovchain (int x, int y, char[][][] data, char[][] level){
        char[] dependence_items = dependencies(x,y,level);
        int n_dependence = dependence_items.length;
        double[] P = new double[n_dependence];
        double P_Total=0;
        int index;
        int n_iterations = data[0][0].length;
        for(int n = 0; n<n_iterations; n++){
            if(Arrays.asList(dependence_items).contains(data[x][y][n])){
                index = Arrays.asList(dependence_items).indexOf(data[x][y][n]);
                P[index]++;
            }
            P_Total++;
        }
        for(int in=0;in<P.length;in++){
          P[in] = P[in] / P_Total;
        }
        return P;
    }
    public double[] Probability_Distribution(char[][][] data){
      long len = MarioLevelModel.getAllTiles().length;
      char[] tiles = MarioLevelModel.getAllTiles();
      double[][][] Prob_dist= new double[rows][columns][len];
      double total = 0;
      for(int x=0; x < rows; x++){
        for(int y=0; y < columns; y++){
          for(int n=0; n < data[0][0].length; n++){
            if(Arrays.asList(tiles).contains(data[x][y][n])){
              Prob_dist[x][y][Arrays.asList(tiles).indexOf(data[x][y][n])]++;
            }
            total++;
          }
        }
      }
      for(int i=0;i<Prob_dist.length;i++){
        Prob_dist[i] = Prob_dist[i]/total; // probably we cant do this in java....gotta run a for loop?
      }
    }
    public double probability_estimation(int x, int y, )//estimation done bruh...in the markovchain
    // did you solve the left right and up case?

    public static void main(String[] args) throws IOException {
       char[][][] traindata = generateTrainData(type,NTD);
       int num_train_levels = NTD;
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

