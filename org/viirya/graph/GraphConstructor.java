
package org.viirya.graph;

import java.io.*;
import java.util.*;

//import java.util.Map;
//import java.util.StringTokenizer;
//import java.util.HashMap;
//import java.util.ArrayList;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.Mapper;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.JobClient;
import org.apache.hadoop.mapred.FileInputFormat;
import org.apache.hadoop.mapred.FileOutputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.io.compress.CompressionCodec;



public class GraphConstructor {

    private static boolean compression = false;

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

    public static class AverageDocumentLengthCalculatorMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, IntWritable, IntWritable> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, IntWritable> output, Reporter reporter) throws NumberFormatException, IOException {
 
            StringTokenizer image_id_tokenizer = tokenize(value, " %");
            if (image_id_tokenizer.countTokens() == 1)
                return;
            String image_features = image_id_tokenizer.nextToken();
            String image_id = image_id_tokenizer.nextToken();

            StringTokenizer features_tokenizer = tokenize(image_features, " :,");
            int number_of_features = features_tokenizer.countTokens() / 2;

            output.collect(new IntWritable(0), new IntWritable(number_of_features));
 
        }

    }
 
    public static class AverageDocumentLengthCalculatorReducer extends SimpleMapReduceBase implements Reducer<IntWritable, IntWritable, IntWritable, Text> {
        public void reduce(IntWritable key, Iterator<IntWritable> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {

            int sum = 0;
            int count = 0;
            while (values.hasNext()) {
                sum += values.next().get(); 
                count++;
            }

            double adl = (double)sum / (double)count;

            output.collect(key, new Text(new Double(adl).toString()));

            try {
                FileSystem fs;
                fs = FileSystem.get(job);
                String path_str = job.get("path");
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

 
    public static class TFCalculatorMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws NumberFormatException, IOException {

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
            double adl = Double.parseDouble(job.get("AverageDocumentLength"));

            IntWritable key_for_image_feature_id = new IntWritable();
            for (int i = 0; i < feature_id.length; i++) {
                if(feature_value[i] > 0) {
                    key_for_image_feature_id.set(feature_id[i]);
 
                    float term_frequency_of_feature = (float)(feature_value[i] / (feature_value[i] + 0.5 + 1.5 * (document_length / adl)));
                    String output_value = "(" + new Float(term_frequency_of_feature).toString() + " " + image_id + ")";
                    output.collect(key_for_image_feature_id, new Text(output_value));
                }
            }

        }
    }

    public static class TFCalculatorReducer extends SimpleMapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
        public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {
        
            StringBuffer strbuf = new StringBuffer();
            strbuf.append("[");
            while (values.hasNext()) {
                strbuf.append(values.next().toString());
            }
            strbuf.append("]");
            output.collect(key, new Text(strbuf.toString()));

        }
    }


    public static class IDFCalculatorMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws NumberFormatException, IOException {

            StringTokenizer tf_values_tokenizer = tokenize(value, " \t()[],");
            String feature_id = tf_values_tokenizer.nextToken();
 
            int number_of_image_tf_pairs = tf_values_tokenizer.countTokens() / 2;
            int number_of_images = Integer.parseInt(job.get("ImagesNumber"));

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
                output.collect(new Text(image_ids[i]), new Text(feature_id_value_pair.toString()));
                feature_id_value_pair.delete(0, feature_id_value_pair.length());
            }

        }
    }

    public static class IDFCalculatorReducer extends SimpleMapReduceBase implements Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

            StringBuffer strbuf = new StringBuffer();
            strbuf.append("[");
            while (values.hasNext()) {
                strbuf.append(values.next().toString());
            }
            strbuf.append("]");
            output.collect(key, new Text(strbuf.toString()));

        }
    }
 
    public static class InvertedListConstructorMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, IntWritable, Text> {
        public void map(LongWritable key, Text value, OutputCollector<IntWritable, Text> output, Reporter reporter) throws NumberFormatException, IOException {

            StringTokenizer image_id_tokenizer = tokenize(value, "\t");
            String image_id = image_id_tokenizer.nextToken();
            String image_features = image_id_tokenizer.nextToken();

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

            IntWritable key_for_image_feature_id = new IntWritable();
            for (int i = 0; i < feature_id.length; i++) {
                if(feature_value[i] > 0) {
                    key_for_image_feature_id.set(feature_id[i]);
                    String output_value = "(" + new Double(feature_value[i]).toString() + " " + new Double(feature_value[i] / square_root).toString() + " " + image_id + ")"; 
                    output.collect(key_for_image_feature_id, new Text(output_value));
                }
            }

        }

    }


    public static class InvertedListConstructorReducer extends SimpleMapReduceBase implements Reducer<IntWritable, Text, IntWritable, Text> {
        public void reduce(IntWritable key, Iterator<Text> values, OutputCollector<IntWritable, Text> output, Reporter reporter) throws IOException {

            StringBuffer strbuf = new StringBuffer();
            strbuf.append("[");
            while (values.hasNext()) {
                strbuf.append(values.next().toString());
            }
            strbuf.append("]");
            output.collect(key, new Text(strbuf.toString()));

        }
    }
 
    public static class PairwiseSimilarityComputationMapper extends SimpleMapReduceBase implements Mapper<LongWritable, Text, Text, DoubleWritable> {
        public void map(LongWritable key, Text value, OutputCollector<Text, DoubleWritable> output, Reporter reporter) throws NumberFormatException, IOException {

            StringTokenizer feature_id_tokenizer = tokenize(value, "\t");
            String feature_id = feature_id_tokenizer.nextToken();
            String image_id_and_feature_value = feature_id_tokenizer.nextToken();

            StringTokenizer image_feature_tokenizer = tokenize(image_id_and_feature_value, " ()[]");
            
            int number_of_images = image_feature_tokenizer.countTokens() / 3;
            String[] image_id = new String[number_of_images];
            double[] feature_tf_idf_value = new double[number_of_images];
            double[] feature_tf_idf_normalized_value = new double[number_of_images];

            int count = 0;
            while (image_feature_tokenizer.hasMoreTokens()) {
                feature_tf_idf_value[count] = Double.parseDouble(image_feature_tokenizer.nextToken());
                feature_tf_idf_normalized_value[count] = Double.parseDouble(image_feature_tokenizer.nextToken());
                image_id[count++] = image_feature_tokenizer.nextToken().toString();
            }


            double threshold = Double.parseDouble(job.get("Threshold"));
        
            Text image_pair = new Text();
            DoubleWritable similarity = new DoubleWritable();
            for(int i = 0; i < image_id.length; i++) {
                for(int j = i + 1; j < image_id.length; j++) {
                    if(feature_tf_idf_value[i] > threshold && feature_tf_idf_value[j] > threshold) {
                        if(image_id[i].compareTo(image_id[j]) < 0) {
                            image_pair.set("(" + image_id[i] + "," + image_id[j] + ")");
                        } else {
                            image_pair.set("(" + image_id[j] + "," + image_id[i] + ")");
                        }
                        similarity.set(feature_tf_idf_normalized_value[i] * feature_tf_idf_normalized_value[j]);        
                        output.collect(image_pair, similarity);
                    }
                }

            }
        }

    }

    public static class PairwiseSimilarityComputationReducer extends SimpleMapReduceBase implements Reducer<Text, DoubleWritable, Text, Text> {
        public void reduce(Text key, Iterator<DoubleWritable> values, OutputCollector<Text, Text> output, Reporter reporter) throws IOException {

            double sum = 0.0d;

            while (values.hasNext()) {
                sum += values.next().get();
            }

            output.collect(key, new Text(new Double(sum).toString()));

        }
    }


    private static void setJobConfCompressed(JobConf job) {
        job.setBoolean("mapred.output.compress", true);
        job.setClass("mapred.output.compression.codec", GzipCodec.class, CompressionCodec.class);
    }


    public static void main(String[] args) throws Exception {

        if (args.length < 2) {
            System.out.println("Usage: GraphConstructor <input path> <threshold> [compress]");    
            System.exit(0);
        }

        if (args.length == 3 && args[2].equals("compress"))
            compression = true;

        Pair adl_and_images_number = calculate_adl(args[0]);
        double adl = ((Double)adl_and_images_number.getFirst()).doubleValue();
        int number_of_images = ((Integer)adl_and_images_number.getSecond()).intValue();
        System.out.println("adl: " + new Double(adl).toString());
        System.out.println("number_of_images: " + new Integer(number_of_images).toString());
        calculate_tf(args[0], adl);
        calculate_idf("output/graph_data/tf", number_of_images);
        construct_inverted_list("output/graph_data/idf");
        compute_pairwise_similarity("output/graph_data/inverted_list", Double.parseDouble(args[1]));

    }
 
    public static Pair calculate_adl(String input_path) throws Exception {

        JobClient job_cli = new JobClient();

        JobConf job_adl_calculator = new JobConf(new Configuration(), GraphConstructor.class);
        job_adl_calculator.setJar("build/GraphConstructor.jar");
        job_adl_calculator.setJobName("AverageDocumentLength_Calculator");

        FileInputFormat.setInputPaths(job_adl_calculator, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_adl_calculator, new Path("output/graph_data/adl"));

        job_adl_calculator.setOutputKeyClass(IntWritable.class);
        job_adl_calculator.setOutputValueClass(Text.class);
        job_adl_calculator.setMapOutputKeyClass(IntWritable.class);
        job_adl_calculator.setMapOutputValueClass(IntWritable.class);
        job_adl_calculator.setMapperClass(AverageDocumentLengthCalculatorMapper.class);
        job_adl_calculator.setReducerClass(AverageDocumentLengthCalculatorReducer.class);
        job_adl_calculator.setNumMapTasks(38);
        job_adl_calculator.setNumReduceTasks(1);
        job_adl_calculator.setLong("dfs.block.size",134217728);
        job_adl_calculator.set("mapred.child.java.opts", "-Xmx2048m");
        job_adl_calculator.set("path", "output/graph_data");

        if (compression)
            setJobConfCompressed(job_adl_calculator);

 
        double adl = 0.0d;
        int number_of_images = 0;
 
        try {
            job_cli.runJob(job_adl_calculator);

            FileSystem fs;
            fs = FileSystem.get(job_adl_calculator);
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

        JobClient job_cli = new JobClient();

        JobConf job_tfidf_tf_calculator = new JobConf(GraphConstructor.class);
        job_tfidf_tf_calculator.setJobName("TFIDF_TF_Calculator");

        FileInputFormat.setInputPaths(job_tfidf_tf_calculator, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_tfidf_tf_calculator, new Path("output/graph_data/tf"));

        job_tfidf_tf_calculator.setOutputKeyClass(IntWritable.class);
        job_tfidf_tf_calculator.setOutputValueClass(Text.class);
        job_tfidf_tf_calculator.setMapperClass(TFCalculatorMapper.class);
        job_tfidf_tf_calculator.setReducerClass(TFCalculatorReducer.class);
        job_tfidf_tf_calculator.setNumMapTasks(38);
        job_tfidf_tf_calculator.setNumReduceTasks(19);
        job_tfidf_tf_calculator.setLong("dfs.block.size",134217728);
        job_tfidf_tf_calculator.set("mapred.child.java.opts", "-Xmx2048m");
        job_tfidf_tf_calculator.set("AverageDocumentLength", new Double(adl).toString());

        if (compression)
            setJobConfCompressed(job_tfidf_tf_calculator);

        try {
            job_cli.runJob(job_tfidf_tf_calculator);
        } catch(Exception e){
            e.printStackTrace();
        }
        
    }
 
    public static void calculate_idf(String input_path, int number_of_images) throws Exception {

        JobClient job_cli = new JobClient();

        JobConf job_tfidf_idf_calculator = new JobConf(GraphConstructor.class);
        job_tfidf_idf_calculator.setJobName("TFIDF_IDF_Calculator");

        FileInputFormat.setInputPaths(job_tfidf_idf_calculator, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_tfidf_idf_calculator, new Path("output/graph_data/idf"));

        job_tfidf_idf_calculator.setOutputKeyClass(Text.class);
        job_tfidf_idf_calculator.setOutputValueClass(Text.class);
        job_tfidf_idf_calculator.setMapperClass(IDFCalculatorMapper.class);
        job_tfidf_idf_calculator.setReducerClass(IDFCalculatorReducer.class);
        job_tfidf_idf_calculator.setNumMapTasks(38);
        job_tfidf_idf_calculator.setNumReduceTasks(19);
        job_tfidf_idf_calculator.setLong("dfs.block.size",134217728);
        job_tfidf_idf_calculator.set("mapred.child.java.opts", "-Xmx2048m");
        job_tfidf_idf_calculator.set("ImagesNumber", new Integer(number_of_images).toString());

        if (compression)
            setJobConfCompressed(job_tfidf_idf_calculator);

        try {
            job_cli.runJob(job_tfidf_idf_calculator);
        } catch(Exception e){
            e.printStackTrace();
        }
        
    }
 
    public static void construct_inverted_list(String input_path) throws Exception {

        JobClient job_cli = new JobClient();

        JobConf job_inverted_list_constructor = new JobConf(GraphConstructor.class);
        job_inverted_list_constructor.setJobName("Inverted List Constructor");

        FileInputFormat.setInputPaths(job_inverted_list_constructor, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_inverted_list_constructor, new Path("output/graph_data/inverted_list"));

        job_inverted_list_constructor.setOutputKeyClass(IntWritable.class);
        job_inverted_list_constructor.setOutputValueClass(Text.class);
        job_inverted_list_constructor.setMapperClass(InvertedListConstructorMapper.class);
        job_inverted_list_constructor.setReducerClass(InvertedListConstructorReducer.class);
        job_inverted_list_constructor.setNumMapTasks(38);
        job_inverted_list_constructor.setNumReduceTasks(19);
        job_inverted_list_constructor.setLong("dfs.block.size",134217728);
        job_inverted_list_constructor.set("mapred.child.java.opts", "-Xmx2048m");

        if (compression)
            setJobConfCompressed(job_inverted_list_constructor);

        try {
            job_cli.runJob(job_inverted_list_constructor);
        } catch(Exception e){
            e.printStackTrace();
        }
        
    }
 
    public static void compute_pairwise_similarity(String input_path, double threshold) throws Exception {

        JobClient job_cli = new JobClient();

        JobConf job_pairwise_similarity_computation = new JobConf(GraphConstructor.class);
        job_pairwise_similarity_computation.setJobName("Computing Pairwise Similarity");

        FileInputFormat.setInputPaths(job_pairwise_similarity_computation, new Path(input_path + "/*"));
        FileOutputFormat.setOutputPath(job_pairwise_similarity_computation, new Path("output/graph_data/graph"));

        job_pairwise_similarity_computation.setOutputKeyClass(Text.class);
        job_pairwise_similarity_computation.setOutputValueClass(Text.class);
        job_pairwise_similarity_computation.setMapOutputKeyClass(Text.class);
        job_pairwise_similarity_computation.setMapOutputValueClass(DoubleWritable.class);
        job_pairwise_similarity_computation.setMapperClass(PairwiseSimilarityComputationMapper.class);
        job_pairwise_similarity_computation.setReducerClass(PairwiseSimilarityComputationReducer.class);
        job_pairwise_similarity_computation.setNumMapTasks(38);
        job_pairwise_similarity_computation.setNumReduceTasks(19);
        job_pairwise_similarity_computation.setLong("dfs.block.size",134217728);
        job_pairwise_similarity_computation.set("mapred.child.java.opts", "-Xmx2048m");
        job_pairwise_similarity_computation.set("Threshold", new Double(threshold).toString());

        if (compression)
            setJobConfCompressed(job_pairwise_similarity_computation);

        try {
            job_cli.runJob(job_pairwise_similarity_computation);
        } catch(Exception e){
            e.printStackTrace();
        }
        
    }
 
}


class Pair {

    public Object o1;
    public Object o2;
    public Pair(Object o1, Object o2) { this.o1 = o1; this.o2 = o2; }
    
    public static boolean same(Object o1, Object o2) {
      return o1 == null ? o2 == null : o1.equals(o2);
    }
    
    Object getFirst() { return o1; }
    Object getSecond() { return o2; }
    
    void setFirst(Object o) { o1 = o; }
    void setSecond(Object o) { o2 = o; }
    
    public boolean equals(Object obj) {
        if( ! (obj instanceof Pair))
            return false;
        Pair p = (Pair)obj;
        return same(p.o1, this.o1) && same(p.o2, this.o2);
    }
    
    public String toString() {
        return "Pair{"+o1+", "+o2+"}";
    }

}
