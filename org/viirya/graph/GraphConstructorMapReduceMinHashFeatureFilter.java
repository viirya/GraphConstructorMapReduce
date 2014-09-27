
package org.viirya.graph;

import java.io.*;
import java.util.*;
import java.lang.reflect.Array;
//import java.util.Map;
//import java.util.StringTokenizer;
//import java.util.HashMap;
//import java.util.ArrayList;

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
//import org.apache.hadoop.mapred.MapReduceBase;
//import org.apache.hadoop.mapred.Mapper;
//import org.apache.hadoop.mapred.OutputCollector;
//import org.apache.hadoop.mapred.Reducer;
//import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.fs.*;
//import org.apache.hadoop.mapred.JobConf;
//import org.apache.hadoop.mapred.JobClient;
//import org.apache.hadoop.mapred.FileInputFormat;
//import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.io.compress.CompressionCodec;


public class GraphConstructorMapReduceMinHashFeatureFilter {

    private static boolean compression = false;
/*
    public static class SimpleMapReduceBase extends MapReduceBase {
        JobConf job;
        @Override
        public void configure(JobConf job) {
            super.configure(job);
            this.job = job;
        }

        public StringTokenizer tokenize(String line, String pattern) {
            StringTokenizer tokenizer = new StringTokenizer(line, pattern);
            return tokenizer;
        } 

        public StringTokenizer tokenize(Text value, String pattern) {
            String line = value.toString();
            StringTokenizer tokenizer = new StringTokenizer(line, pattern);
            return tokenizer;
        }
    }
*/

    public static StringTokenizer tokenize(String line, String pattern) {
        StringTokenizer tokenizer = new StringTokenizer(line, pattern);
        return tokenizer;
    } 

    public static StringTokenizer tokenize(Text value, String pattern) {
        String line = value.toString();
        StringTokenizer tokenizer = new StringTokenizer(line, pattern);
        return tokenizer;
    }


    public static class AverageDocumentLengthCalculatorMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {
 
            StringTokenizer image_id_tokenizer = tokenize(value, " %");
            if (image_id_tokenizer.countTokens() == 1)
                return;
            String image_features = image_id_tokenizer.nextToken();
            String image_id = image_id_tokenizer.nextToken();

            StringTokenizer features_tokenizer = tokenize(image_features, " :,");
            int number_of_features = features_tokenizer.countTokens() / 2;

            context.write(new IntWritable(0), new IntWritable(number_of_features));
 
        }

    }
 
    public static class AverageDocumentLengthCalculatorReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {
        public void reduce(IntWritable key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {

            int sum = 0;
            int count = 0;
            for (IntWritable val: values) {
                sum += val.get(); 
                count++;
            }

            double adl = (double)sum / (double)count;

            context.write(key, new Text(new Double(adl).toString()));

            try {
                FileSystem fs;
                fs = FileSystem.get(context.getConfiguration());
                String path_str = context.getConfiguration().get("path");
                Path path_adl_output = new Path(path_str + "/adl_output");
                if(!fs.exists(path_adl_output)) {
                    DataOutputStream out = fs.create(path_adl_output);
                    out.writeDouble(adl);
                    out.close();
                }

                Path path_images_number_output = new Path(path_str + "/images_number_output");
                if(!fs.exists(path_images_number_output)) {
                    DataOutputStream out = fs.create(path_images_number_output);
                    out.writeInt(count);
                    out.close();
                }
            
            } catch(Exception e) {
                throw new IOException(e.getMessage());
            }
 
        }
    }

 
    public static class TFCalculatorMapper extends Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            StringTokenizer image_id_tokenizer = tokenize(value, " %");
            if (image_id_tokenizer.countTokens() == 1)
                return;
            String image_features = image_id_tokenizer.nextToken();
            String image_id = image_id_tokenizer.nextToken();

            StringTokenizer features_tokenizer = tokenize(image_features, " :,");
            int number_of_features = features_tokenizer.countTokens() / 2;
            int document_length = 0;
            double square_sum = 0;
            int[] feature_id = new int[number_of_features];
            double[] feature_value = new double[number_of_features];

            int count = 0;
            while (features_tokenizer.hasMoreTokens()) {
                feature_id[count] = Integer.parseInt(features_tokenizer.nextToken());
                feature_value[count] = Double.parseDouble(features_tokenizer.nextToken());
                document_length++;
                square_sum += feature_value[count] * feature_value[count];
                count++;
            }

            double square_root = Math.sqrt(square_sum);
            double adl = Double.parseDouble(context.getConfiguration().get("AverageDocumentLength"));

            IntWritable key_for_image_feature_id = new IntWritable();
            for (int i = 0; i < feature_id.length; i++) {
                if(feature_value[i] > 0) {
                    key_for_image_feature_id.set(feature_id[i]);
 
                    float term_frequency_of_feature = (float)(feature_value[i] / (feature_value[i] + 0.5 + 1.5 * (document_length / adl)));
                    String output_value = "(" + new Float(term_frequency_of_feature).toString() + " " + image_id + ")";
                    context.write(key_for_image_feature_id, new Text(output_value));
                }
            }

        }
    }

    public static class TFCalculatorReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        
            StringBuffer strbuf = new StringBuffer();
            strbuf.append("[");
            for (Text val: values) {
                strbuf.append(val.toString());
            }
            strbuf.append("]");
            context.write(key, new Text(strbuf.toString()));

        }
    }


    public static class IDFCalculatorMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            StringTokenizer tf_values_tokenizer = tokenize(value, " \t()[],");
            String feature_id = tf_values_tokenizer.nextToken();
 
            int number_of_image_tf_pairs = tf_values_tokenizer.countTokens() / 2;
            int number_of_images = Integer.parseInt(context.getConfiguration().get("ImagesNumber"));

            float[] tf_values = new float[number_of_image_tf_pairs];
            String[] image_ids = new String[number_of_image_tf_pairs];
       
            int count = 0;
            while (tf_values_tokenizer.hasMoreTokens()) {
                tf_values[count] =  Float.parseFloat(tf_values_tokenizer.nextToken());
                if (tf_values_tokenizer.hasMoreTokens()) {
                    image_ids[count++] = tf_values_tokenizer.nextToken();
                }
            } 

            double idf = Math.log((number_of_images + 0.5) / number_of_image_tf_pairs) / Math.log(number_of_images + 1.0);
            double tfidf_value = 0.0d;

            StringBuffer feature_id_value_pair = new StringBuffer();

            for(int i = 0; i < tf_values.length; i++) {
                tfidf_value = tf_values[i] * idf;
                //String feature_id_value_pair = "(" + feature_id + " " + new Float(tfidf_value).toString() + ")";
                feature_id_value_pair = feature_id_value_pair.append("(").append(feature_id).append(" ").append(tfidf_value).append(")");
                context.write(new Text(image_ids[i]), new Text(feature_id_value_pair.toString()));
                feature_id_value_pair.delete(0, feature_id_value_pair.length());
            }

        }
    }

    public static class IDFCalculatorReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            StringBuffer strbuf = new StringBuffer();
            strbuf.append("[");
            for (Text val: values) {
                strbuf.append(val.toString());
            }
            strbuf.append("]");
            context.write(key, new Text(strbuf.toString()));

        }
    }
 
    public static class AddMinHashInfoMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            int min_cluster_size = Integer.parseInt(context.getConfiguration().get("MinClusterSize"));

            StringTokenizer image_id_tokenizer = tokenize(value, "\t");
            String image_id_or_cluster_id = image_id_tokenizer.nextToken();
            String image_features_or_image_ids = image_id_tokenizer.nextToken();

            StringTokenizer test_tokenizer = tokenize(image_id_or_cluster_id, ":");
            /* cluster */
            if (test_tokenizer.countTokens() > 1) {
                //String cluster_id = image_id_or_cluster_id;    
                test_tokenizer.nextToken();
                String cluster_id = test_tokenizer.nextToken();
                String[] image_ids = image_features_or_image_ids.split(" ");
                Text image_pair = new Text();
                if (image_ids.length > min_cluster_size) {
                    for (int i = 0; i < image_ids.length; i++) {
                        for(int j = i + 1; j < image_ids.length; j++) {
                            if(image_ids[i].compareTo(image_ids[j]) < 0) {
                                image_pair.set("(" + image_ids[i] + "," + image_ids[j] + ")");
                            } else {
                                image_pair.set("(" + image_ids[j] + "," + image_ids[i] + ")");
                            }
                            context.write(image_pair, new Text("1"));
                        }
                        //context.write(new Text(image_ids[i]), new Text("c%" + cluster_id));
                    }
                }
            } 
            //else {
            //    context.write(new Text(image_id_or_cluster_id), new Text(image_features_or_image_ids));
            //}

        }

    }


    public static class AddMinHashInfoReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            String features = null;
            String cluster_info = null;
            for (Text val: values) {
                StringTokenizer test_tokenizer = tokenize(val, "%");
                if (test_tokenizer.countTokens() > 1) {

                    test_tokenizer.nextToken();
                    if (cluster_info == null)
                        cluster_info = test_tokenizer.nextToken();                    
                    else                        
                        cluster_info = cluster_info + ":" + test_tokenizer.nextToken();

                    //test_tokenizer.nextToken();
                    //cluster_info = test_tokenizer.nextToken();
                    
                    //String cluster = test_tokenizer.nextToken();

                    //test_tokenizer = tokenize(cluster, ":");
                    //test_tokenizer.nextToken();
                    //cluster = test_tokenizer.nextToken();
                    //cluster_info = cluster;    
                } else {
                    features = val.toString();    
                }
            }
            if (cluster_info != null && features != null)
                context.write(new Text(key.toString() + "," + cluster_info), new Text(features));

        }
    }
 
    public static class InvertedListConstructorMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            StringTokenizer image_id_tokenizer = tokenize(value, "\t");
            String image_id = image_id_tokenizer.nextToken();
            String image_features = image_id_tokenizer.nextToken();

            //StringTokenizer cluster_tokenizer = tokenize(image_id_and_cluster_id, ",");
            //String image_id = cluster_tokenizer.nextToken();
            //String cluster_id = cluster_tokenizer.nextToken();

            StringTokenizer features_tokenizer = tokenize(image_features, " ()[]:,");
            int number_of_features = features_tokenizer.countTokens() / 2;
            double square_sum = 0;
            int[] feature_id = new int[number_of_features];
            double[] feature_value = new double[number_of_features];

            int count = 0;
            while (features_tokenizer.hasMoreTokens()) {
                feature_id[count] = Integer.parseInt(features_tokenizer.nextToken());
                feature_value[count] = Double.parseDouble(features_tokenizer.nextToken());
                square_sum += feature_value[count] * feature_value[count];
                count++;
            }

            double square_root = Math.sqrt(square_sum);

            Text key_for_image_feature_id = new Text();
            for (int i = 0; i < feature_id.length; i++) {
                if(feature_value[i] > 0) {
                    //String[] cluster_ids = cluster_id.split(":");

                    //for (int j = 0; j < cluster_ids.length; j++) {
                    key_for_image_feature_id.set(new Integer(feature_id[i]).toString());
                    String output_value = "(" + new Double(feature_value[i]).toString() + " " + new Double(feature_value[i] / square_root).toString() + " " + image_id + ")"; 
                    context.write(key_for_image_feature_id, new Text(output_value));
                    //}
                }
            }

        }

    }


    public static class InvertedListConstructorReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            StringBuffer strbuf = new StringBuffer();
            strbuf.append("[");
            for (Text val: values) {
                strbuf.append(val.toString());
            }
            strbuf.append("]");
            context.write(key, new Text(strbuf.toString()));

        }
    }
 
    public static class PrePairwiseSimilarityComputationMapper extends Mapper<LongWritable, Text, Text, FloatArrayWritable> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            StringTokenizer feature_id_tokenizer = tokenize(value, "\t");
            String feature_id = feature_id_tokenizer.nextToken();
            String image_id_and_feature_value = feature_id_tokenizer.nextToken();

            if (image_id_and_feature_value.equals("1")) {
                // image pair
                FloatArrayWritable fake_float_pairs = new FloatArrayWritable();
                FloatWritable[] pair = new FloatWritable[2];
                pair[0] = new FloatWritable();
                pair[1] = new FloatWritable();
                pair[0].set(-1.0f);
                pair[1].set(-1.0f);
                fake_float_pairs.set(pair);
                context.write(new Text(feature_id), fake_float_pairs);
            } else {

                 //String[] feature_id_and_cluster_id = feature_id.split(",");
                
                StringTokenizer image_feature_tokenizer = tokenize(image_id_and_feature_value, " ()[]");
                
                int number_of_images = image_feature_tokenizer.countTokens() / 3;
                String[] image_id = new String[number_of_images];
                String[] feature_tf_idf_value = new String[number_of_images];
                String[] feature_tf_idf_normalized_value = new String[number_of_images];
                
                int count = 0;
                while (image_feature_tokenizer.hasMoreTokens()) {
                    feature_tf_idf_value[count] = image_feature_tokenizer.nextToken();
                    feature_tf_idf_normalized_value[count] = image_feature_tokenizer.nextToken();
                    image_id[count++] = image_feature_tokenizer.nextToken().toString();
                }

                float threshold = Float.parseFloat(context.getConfiguration().get("Threshold"));                
                Text image_pair = new Text();
                FloatArrayWritable feature_values = new FloatArrayWritable();
                FloatWritable[] normalized_values = new FloatWritable[2];
                normalized_values[0] = new FloatWritable();
                normalized_values[1] = new FloatWritable();
                for(int i = 0; i < image_id.length; i++) {
                    for(int j = i + 1; j < image_id.length; j++) {
                        float l_raw_value = Float.parseFloat(feature_tf_idf_value[i]);
                        float r_raw_value = Float.parseFloat(feature_tf_idf_value[j]);
                        float l_value = Float.parseFloat(feature_tf_idf_normalized_value[i]);
                        float r_value = Float.parseFloat(feature_tf_idf_normalized_value[j]);

                        if (l_raw_value > threshold && r_raw_value > threshold) {
                            if(image_id[i].compareTo(image_id[j]) < 0) {
                                image_pair.set("(" + image_id[i] + "," + image_id[j] + ")");
                            } else {
                                image_pair.set("(" + image_id[j] + "," + image_id[i] + ")");
                            }
                            normalized_values[0].set(l_value);
                            normalized_values[1].set(r_value);
                            feature_values.set(normalized_values);        
                            context.write(image_pair, feature_values);
                        }
                    }
                
                }
 
            }
        }
    }
 
    public static class PrePairwiseSimilarityComputationCombiner extends Reducer<Text, FloatArrayWritable, Text, FloatArrayWritable> {
        public void reduce(Text key, Iterable<FloatArrayWritable> values, Context context) throws IOException, InterruptedException {

            boolean in_min_hash_bin = false;
            HashMap<Integer, Float> first_feature_value = new HashMap<Integer, Float>();
            HashMap<Integer, Float> second_feature_value = new HashMap<Integer, Float>();

            int count = 0; 
            for (FloatArrayWritable val: values) {
                Object pair = val.toArray();
                
                if (((FloatWritable)Array.get(pair, 0)).get() == -1.0f && ((FloatWritable)Array.get(pair, 1)).get() == -1.0f) {
                    context.write(key, val);
                    in_min_hash_bin = true;
                } else if (((FloatWritable)Array.get(pair, 0)).get() != -1.0f && ((FloatWritable)Array.get(pair, 1)).get() == -1.0f ) {                    
                    first_feature_value.put(count, ((FloatWritable)Array.get(pair, 0)).get());
                    second_feature_value.put(count++, 1.0f);
                } else {
                    first_feature_value.put(count, ((FloatWritable)Array.get(pair, 0)).get());
                    second_feature_value.put(count++, ((FloatWritable)Array.get(pair, 1)).get());
                }
            }

            float threshold = Float.parseFloat(context.getConfiguration().get("Threshold"));
            float similarity = 0.0f;

            FloatArrayWritable feature_values = new FloatArrayWritable();                
            FloatWritable[] normalized_values = new FloatWritable[2];
            normalized_values[0] = new FloatWritable();
            normalized_values[1] = new FloatWritable();

            //if (in_min_hash_bin) {
                for(Map.Entry<Integer, Float> entry : first_feature_value.entrySet()) {
                    Integer feature_key = entry.getKey();
                    float f_value = first_feature_value.get(feature_key);
                    float s_value = second_feature_value.get(feature_key);
                    if (f_value > threshold && s_value > threshold)
                        similarity += first_feature_value.get(feature_key) * second_feature_value.get(feature_key);
                }
                if (similarity > 0.0d) {
                    normalized_values[0].set(similarity);                            
                    normalized_values[1].set(-1.0f);
                    feature_values.set(normalized_values);                    
                    context.write(key, feature_values);
                }
            //}
        }
    }
 
    public static class PrePairwiseSimilarityComputationReducer extends Reducer<Text, FloatArrayWritable, Text, Text> {
        public void reduce(Text key, Iterable<FloatArrayWritable> values, Context context) throws IOException, InterruptedException {

            boolean in_min_hash_bin = false;
            HashMap<Integer, Float> first_feature_value = new HashMap<Integer, Float>();
            HashMap<Integer, Float> second_feature_value = new HashMap<Integer, Float>();

            int count = 0; 
            for (FloatArrayWritable val: values) {
                Object pair = val.toArray();
                
                if (((FloatWritable)Array.get(pair, 0)).get() == -1.0f && ((FloatWritable)Array.get(pair, 1)).get() == -1.0f) {
                    in_min_hash_bin = true;
                } else if (((FloatWritable)Array.get(pair, 0)).get() != -1.0f && ((FloatWritable)Array.get(pair, 1)).get() == -1.0f ) {
                    first_feature_value.put(count, ((FloatWritable)Array.get(pair, 0)).get());
                    second_feature_value.put(count++, 1.0f);
                } else {
                    //String[] feature_values = val.toString().split(",");
                    //first_feature_value.put(count, Float.parseFloat(feature_values[0])); 
                    //second_feature_value.put(count++, Float.parseFloat(feature_values[1])); 
                    first_feature_value.put(count, ((FloatWritable)Array.get(pair, 0)).get());
                    second_feature_value.put(count++, ((FloatWritable)Array.get(pair, 1)).get());
                }
            }

            float threshold = Float.parseFloat(context.getConfiguration().get("Threshold"));
            float similarity = 0.0f;
            if (in_min_hash_bin) {
                for(Map.Entry<Integer, Float> entry : first_feature_value.entrySet()) {
                    Integer feature_key = entry.getKey();
                    float f_value = first_feature_value.get(feature_key);
                    float s_value = second_feature_value.get(feature_key);
                    if (f_value > threshold && s_value > threshold)
                        similarity += first_feature_value.get(feature_key) * second_feature_value.get(feature_key);
                }
                if (similarity > 0.0f)
                    context.write(key, new Text(new Float(similarity).toString()));
            }
        }
    }

 
    public static class PairwiseSimilarityComputationMapper extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws NumberFormatException, IOException, InterruptedException {

            StringTokenizer image_pair_tokenizer = tokenize(value, "\t");
            context.write(new Text(image_pair_tokenizer.nextToken()), new Text(image_pair_tokenizer.nextToken()));

        }

    }

    public static class PairwiseSimilarityComputationReducer extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {

            float sum = 0.0f;

            for (Text val: values) {
                float similarity = Float.parseFloat(val.toString()); 
                sum += similarity;
            }

            context.write(key, new Text(new Float(sum).toString()));

        }
    }


    private static void setJobConfCompressed(Configuration jobconf) {
        jobconf.setBoolean("mapred.output.compress", true);
        jobconf.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    }

    private static void deletePathOnHDFS(Configuration conf, String path) {
        try {
            FileSystem fs;
            fs = FileSystem.get(conf);
            Path path_on_hdfs = new Path(path);
            if(fs.exists(path_on_hdfs)){
                fs.delete(path_on_hdfs, true);
            }
        } catch(Exception e){
            e.printStackTrace();
        }
    }


    public static void main(String[] args) throws Exception {

        if (args.length < 4) {
            System.out.println("Usage: GraphConstructorMapReduceMinHashFeatureFilter <input path> <cluster path> <threshold> <cluster min size> [compress]");    
            System.exit(0);
        }

        if (args.length == 5 && args[4].equals("compress"))
            compression = true;

        int min_cluster_size = Integer.parseInt(args[3]);

        Pair adl_and_images_number = calculate_adl(args[0]);
        double adl = ((Double)adl_and_images_number.getFirst()).doubleValue();
        int number_of_images = ((Integer)adl_and_images_number.getSecond()).intValue();
        System.out.println("adl: " + new Double(adl).toString());
        System.out.println("number_of_images: " + new Integer(number_of_images).toString());
        calculate_tf(args[0], adl);
        calculate_idf("output/graph_data/tf", number_of_images);
        addminhashinfo(args[1], min_cluster_size);
        construct_inverted_list("output/graph_data/idf");
        pre_compute_pairwise_similarity("output/graph_data/inverted_list", "output/graph_data/image_pairs", Float.parseFloat(args[2]));
        //compute_pairwise_similarity("output/graph_data/feature_values");

    }
 
    public static Pair calculate_adl(String input_path) throws Exception {

        //JobClient job_cli = new JobClient();
        //JobConf job_adl_calculator = new JobConf(new Configuration(), GraphConstructor.class);

        Configuration conf = new Configuration();
        conf.set("path", "output/graph_data");
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        if (compression)
            setJobConfCompressed(conf);        

        Job job_adl_calculator = new Job(conf);
        job_adl_calculator.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_adl_calculator.setJobName("AverageDocumentLength_Calculator");

        FileInputFormat.setInputPaths(job_adl_calculator, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_adl_calculator, new Path("output/graph_data/adl"));

        job_adl_calculator.setOutputKeyClass(IntWritable.class);
        job_adl_calculator.setOutputValueClass(Text.class);
        job_adl_calculator.setMapOutputKeyClass(IntWritable.class);
        job_adl_calculator.setMapOutputValueClass(IntWritable.class);
        job_adl_calculator.setMapperClass(AverageDocumentLengthCalculatorMapper.class);
        //job_adl_calculator.setCombinerClass(AverageDocumentLengthCalculatorReducer.class);
        job_adl_calculator.setReducerClass(AverageDocumentLengthCalculatorReducer.class);
        //job_adl_calculator.setNumMapTasks(38);
        job_adl_calculator.setNumReduceTasks(1);
        //job_adl_calculator.setLong("dfs.block.size",134217728);
        //job_adl_calculator.set("mapred.child.java.opts", "-Xmx2048m");
        //job_adl_calculator.set("path", "output/graph_data");

        double adl = 0.0d;
        int number_of_images = 0;
 
        try {
            //job_cli.runJob(job_adl_calculator);

            job_adl_calculator.waitForCompletion(true);

            FileSystem fs;
            fs = FileSystem.get(conf);
            String path_str = "output/graph_data";
            Path path_adl_output = new Path(path_str + "/adl_output");
            if(fs.exists(path_adl_output)) {
              DataInputStream in = fs.open(path_adl_output);
              adl = in.readDouble();
              in.close();
            }

            Path path_images_number_output = new Path(path_str + "/images_number_output");
            if(fs.exists(path_images_number_output)) {
              DataInputStream in = fs.open(path_images_number_output);
              number_of_images = in.readInt();
              in.close();
            }
 
        } catch(Exception e){
            e.printStackTrace();
        }

        return new Pair(new Double(adl), new Integer(number_of_images));

    }


 
    public static void calculate_tf(String input_path, double adl) throws Exception {

        //JobClient job_cli = new JobClient();
        //JobConf job_tfidf_tf_calculator = new JobConf(GraphConstructor.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        conf.set("AverageDocumentLength", new Double(adl).toString());
        if (compression)
            setJobConfCompressed(conf);
        
        Job job_tfidf_tf_calculator = new Job(conf);
        job_tfidf_tf_calculator.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_tfidf_tf_calculator.setJobName("TFIDF_TF_Calculator");

        FileInputFormat.setInputPaths(job_tfidf_tf_calculator, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_tfidf_tf_calculator, new Path("output/graph_data/tf"));

        job_tfidf_tf_calculator.setOutputKeyClass(IntWritable.class);
        job_tfidf_tf_calculator.setOutputValueClass(Text.class);
        job_tfidf_tf_calculator.setMapperClass(TFCalculatorMapper.class);
        //job_tfidf_tf_calculator.setCombinerClass(TFCalculatorReducer.class);
        job_tfidf_tf_calculator.setReducerClass(TFCalculatorReducer.class);
        //job_tfidf_tf_calculator.setNumMapTasks(38);
        job_tfidf_tf_calculator.setNumReduceTasks(19);
        //job_tfidf_tf_calculator.setLong("dfs.block.size",134217728);
        //job_tfidf_tf_calculator.set("mapred.child.java.opts", "-Xmx2048m");
        //job_tfidf_tf_calculator.set("AverageDocumentLength", new Double(adl).toString());

        try {
            //job_cli.runJob(job_tfidf_tf_calculator);
            job_tfidf_tf_calculator.waitForCompletion(true);            
        } catch(Exception e){
            e.printStackTrace();
        }
        
    }
 
    public static void calculate_idf(String input_path, int number_of_images) throws Exception {

        //JobClient job_cli = new JobClient();
        //JobConf job_tfidf_idf_calculator = new JobConf(GraphConstructor.class);
        
        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        conf.set("ImagesNumber", new Integer(number_of_images).toString());
        if (compression)
            setJobConfCompressed(conf);

        Job job_tfidf_idf_calculator = new Job(conf);
        job_tfidf_idf_calculator.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_tfidf_idf_calculator.setJobName("TFIDF_IDF_Calculator");

        FileInputFormat.setInputPaths(job_tfidf_idf_calculator, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_tfidf_idf_calculator, new Path("output/graph_data/idf"));

        job_tfidf_idf_calculator.setOutputKeyClass(Text.class);
        job_tfidf_idf_calculator.setOutputValueClass(Text.class);
        job_tfidf_idf_calculator.setMapperClass(IDFCalculatorMapper.class);
        //job_tfidf_idf_calculator.setCombinerClass(IDFCalculatorReducer.class);
        job_tfidf_idf_calculator.setReducerClass(IDFCalculatorReducer.class);
        //job_tfidf_idf_calculator.setNumMapTasks(38);
        job_tfidf_idf_calculator.setNumReduceTasks(19);
        //job_tfidf_idf_calculator.setLong("dfs.block.size",134217728);
        //job_tfidf_idf_calculator.set("mapred.child.java.opts", "-Xmx2048m");
        //job_tfidf_idf_calculator.set("ImagesNumber", new Integer(number_of_images).toString());

        try {
            //job_cli.runJob(job_tfidf_idf_calculator);
            job_tfidf_idf_calculator.waitForCompletion(true);
        } catch(Exception e){
            e.printStackTrace();
        }

        deletePathOnHDFS(conf, input_path);
        
    }
 
    public static void addminhashinfo(String cluster_path, int min_cluster_size) throws Exception {

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        conf.set("MinClusterSize", new Integer(min_cluster_size).toString());
        if (compression)
            setJobConfCompressed(conf);

        Job job_addminhashinfo = new Job(conf);
        job_addminhashinfo.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_addminhashinfo.setJobName("Add MinHash Information");

        FileInputFormat.setInputPaths(job_addminhashinfo, new Path(cluster_path + "/*"));
        FileOutputFormat.setOutputPath(job_addminhashinfo, new Path("output/graph_data/image_pairs"));

        job_addminhashinfo.setOutputKeyClass(Text.class);
        job_addminhashinfo.setOutputValueClass(Text.class);
        job_addminhashinfo.setMapperClass(AddMinHashInfoMapper.class);
        //job_addminhashinfo.setCombinerClass(AddMinHashInfoReducer.class);
        //job_addminhashinfo.setReducerClass(AddMinHashInfoReducer.class);
        job_addminhashinfo.setNumReduceTasks(19);


        try {
            job_addminhashinfo.waitForCompletion(true);
        } catch(Exception e){
            e.printStackTrace();
        }

        //deletePathOnHDFS(conf, input_path);
        
    }
 
    public static void construct_inverted_list(String input_path) throws Exception {

        //JobClient job_cli = new JobClient();
        //JobConf job_inverted_list_constructor = new JobConf(GraphConstructor.class);
        
        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        if (compression)
            setJobConfCompressed(conf);

        Job job_inverted_list_constructor = new Job(conf);
        job_inverted_list_constructor.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_inverted_list_constructor.setJobName("Inverted List Constructor");

        FileInputFormat.setInputPaths(job_inverted_list_constructor, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_inverted_list_constructor, new Path("output/graph_data/inverted_list"));

        job_inverted_list_constructor.setOutputKeyClass(Text.class);
        job_inverted_list_constructor.setOutputValueClass(Text.class);
        job_inverted_list_constructor.setMapperClass(InvertedListConstructorMapper.class);
        //job_inverted_list_constructor.setCombinerClass(InvertedListConstructorReducer.class);
        job_inverted_list_constructor.setReducerClass(InvertedListConstructorReducer.class);
        //job_inverted_list_constructor.setNumMapTasks(38);
        job_inverted_list_constructor.setNumReduceTasks(19);
        //job_inverted_list_constructor.setLong("dfs.block.size",134217728);
        //job_inverted_list_constructor.set("mapred.child.java.opts", "-Xmx2048m");


        try {
            //job_cli.runJob(job_inverted_list_constructor);
            job_inverted_list_constructor.waitForCompletion(true);
        } catch(Exception e){
            e.printStackTrace();
        }

        deletePathOnHDFS(conf, input_path);
        
    }
 
    public static void pre_compute_pairwise_similarity(String input_path, String image_pairs_path, float threshold) throws Exception {

        //JobClient job_cli = new JobClient();
        //JobConf job_pairwise_similarity_computation = new JobConf(GraphConstructor.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        conf.set("Threshold", new Float(threshold).toString());
        if (compression)
            setJobConfCompressed(conf);

        Job job_pairwise_similarity_computation = new Job(conf);
        job_pairwise_similarity_computation.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_pairwise_similarity_computation.setJobName("Pre Computing Pairwise Similarity");

        FileInputFormat.setInputPaths(job_pairwise_similarity_computation, new Path(input_path + "/*"), new Path(image_pairs_path + "/*"));
        FileOutputFormat.setOutputPath(job_pairwise_similarity_computation, new Path("output/graph_data/feature_values"));

        job_pairwise_similarity_computation.setOutputKeyClass(Text.class);
        job_pairwise_similarity_computation.setOutputValueClass(Text.class);
        job_pairwise_similarity_computation.setMapOutputKeyClass(Text.class);
        job_pairwise_similarity_computation.setMapOutputValueClass(FloatArrayWritable.class);
        job_pairwise_similarity_computation.setMapperClass(PrePairwiseSimilarityComputationMapper.class);
        job_pairwise_similarity_computation.setCombinerClass(PrePairwiseSimilarityComputationCombiner.class);
        job_pairwise_similarity_computation.setReducerClass(PrePairwiseSimilarityComputationReducer.class);
        //job_pairwise_similarity_computation.setNumMapTasks(38);
        job_pairwise_similarity_computation.setNumReduceTasks(19);
        //job_pairwise_similarity_computation.setLong("dfs.block.size",134217728);
        //job_pairwise_similarity_computation.set("mapred.child.java.opts", "-Xmx2048m");
        //job_pairwise_similarity_computation.set("Threshold", new Float(threshold).toString());

        try {
            //job_cli.runJob(job_pairwise_similarity_computation);
            job_pairwise_similarity_computation.waitForCompletion(true);
        } catch(Exception e){
            e.printStackTrace();
        }

        deletePathOnHDFS(conf, input_path);
        
    }
 
    public static void compute_pairwise_similarity(String input_path) throws Exception {

        //JobClient job_cli = new JobClient();
        //JobConf job_pairwise_similarity_computation = new JobConf(GraphConstructor.class);

        Configuration conf = new Configuration();
        conf.setLong("dfs.block.size",134217728);
        conf.set("mapred.child.java.opts", "-Xmx2048m");
        if (compression)
            setJobConfCompressed(conf);

        Job job_pairwise_similarity_computation = new Job(conf);
        job_pairwise_similarity_computation.setJarByClass(GraphConstructorMapReduceMinHashFeatureFilter.class);
        job_pairwise_similarity_computation.setJobName("Computing Pairwise Similarity");

        FileInputFormat.setInputPaths(job_pairwise_similarity_computation, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_pairwise_similarity_computation, new Path("output/graph_data/graph"));

        job_pairwise_similarity_computation.setOutputKeyClass(Text.class);
        job_pairwise_similarity_computation.setOutputValueClass(Text.class);
        job_pairwise_similarity_computation.setMapOutputKeyClass(Text.class);
        job_pairwise_similarity_computation.setMapOutputValueClass(Text.class);
        job_pairwise_similarity_computation.setMapperClass(PairwiseSimilarityComputationMapper.class);
        //job_pairwise_similarity_computation.setCombinerClass(PairwiseSimilarityComputationReducer.class);
        job_pairwise_similarity_computation.setReducerClass(PairwiseSimilarityComputationReducer.class);
        //job_pairwise_similarity_computation.setNumMapTasks(38);
        job_pairwise_similarity_computation.setNumReduceTasks(19);
        //job_pairwise_similarity_computation.setLong("dfs.block.size",134217728);
        //job_pairwise_similarity_computation.set("mapred.child.java.opts", "-Xmx2048m");
        //job_pairwise_similarity_computation.set("Threshold", new Float(threshold).toString());

        try {
            //job_cli.runJob(job_pairwise_similarity_computation);
            job_pairwise_similarity_computation.waitForCompletion(true);
        } catch(Exception e){
            e.printStackTrace();
        }

        deletePathOnHDFS(conf, input_path);
        
    }
 
}

