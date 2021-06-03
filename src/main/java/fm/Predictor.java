package fm;

import org.apache.commons.io.IOUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import redis.clients.jedis.Jedis;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;

/**
 * @author wangke
 * @version 1.0
 * @date 2021/6/3 8:45 下午
 */
public class Predictor {
    private Session sess;
    // private Jedis jedis;

    public void init(String pbFile) throws IOException {
        // jedis = new Jedis("localhost");
        
        Graph graph = new Graph();
        // 这个graphBytes也可以从redis中读取
        // byte[] graphBytes = jedis.get(("fm_model").getBytes());
        byte[] graphBytes = IOUtils.toByteArray(new FileInputStream(pbFile));
        graph.importGraphDef(graphBytes);
        sess = new Session(graph);
    }

    public void inference(float[][] inputFeature) {
        long start = System.currentTimeMillis();
        Tensor embeddingTensor = Tensor.create(inputFeature);
        Tensor rlt = sess.runner().feed("x:0", embeddingTensor).fetch("Identity:0").run().get(0);
        float[][] res = new float[1][(int) rlt.shape()[1]];
        rlt.copyTo(res);
        System.err.println(Arrays.deepToString(res));
        System.out.println("FM time: " + (System.currentTimeMillis() - start) + "ms");
    }

    public static void main(String[] args) throws IOException {
        Predictor predictor = new Predictor();
        predictor.init("src/main/resources/fm/model.pb");

        float[][] input1 = {{0.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, -0.4145451f, -0.35646869f}};
        float[][] input2 = {{0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.53977277f, -0.05042144f}};

        predictor.inference(input1);
        predictor.inference(input2);

    }
}
