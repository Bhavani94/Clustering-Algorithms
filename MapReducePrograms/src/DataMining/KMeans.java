package DataMining;

import java.io.IOException;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeans {

	public static Map<Double, List<Double>> inputData = new HashMap<>();
	public static Map<Double, List<Double>> clusterData = new HashMap<>();
	public static Map<Double, List<Double>> clustRefMap = new HashMap<>();
	public static boolean isMatch = false;
	public static Map<Double, Double> groundTruth = new HashMap<>();
	public static Map<Double, List<Double>> prevClusterList;

	// returns initial clusters<clusterId, coordinates>
	public Map<Double, List<Double>> clusterPoints() throws IOException, InterruptedException {

		// int ktemp[] = new int[k];
		// Random rand = new Random();

		List<Double> centroids = new ArrayList<>(Arrays.asList(3.0,44.0,55.0,11.0,66.0));
		Collections.sort(centroids);

		for (int i = 0; i < centroids.size(); i++) {
			clusterData.put(centroids.get(i), inputData.get(centroids.get(i)));
		}
		return clusterData;
	}

	public Map<Double, List<Double>> updateClusterRef() throws IOException, InterruptedException {
		clustRefMap = new HashMap<>();
		Double dis, ipkey, clkey = 0.0;
		for (Double inputDataKey : inputData.keySet()) {
			List<Double> temp = new ArrayList<>();
			dis = Double.MAX_VALUE;
			ipkey = inputDataKey;
			for (Double clusterDataKey : prevClusterList.keySet()) {
				// find the closest centroid and update the cluster
				// System.out.println(clusterDataKey);
				if (eucDistance(inputData.get(inputDataKey), prevClusterList.get(clusterDataKey)) < dis) {
					dis = eucDistance(inputData.get(inputDataKey), prevClusterList.get(clusterDataKey));
					clkey = clusterDataKey;
				}
			}
			// System.out.println(clkey);
			if (clustRefMap.get(clkey) == null) {
				temp.add(ipkey);
				clustRefMap.put(clkey, temp);
			} else {
				temp = clustRefMap.get(clkey);
				temp.add(ipkey);
				clustRefMap.put(clkey, temp);

			}
		}
		return clustRefMap;
	}

	public Map<Double, List<Double>> updateClusterPoints() throws IOException, InterruptedException {

		for (Double clustRefkey : clustRefMap.keySet()) {
			List<Double> ref = clustRefMap.get(clustRefkey);
			List<Double> sumValues = new ArrayList<Double>(Collections.nCopies(60, 0.0));
			List<Double> sumValuesbyn = new ArrayList<Double>();
			Double sumKey = 0.0;
			for (int i = 0; i < ref.size(); i++) {
				sumKey += ref.get(i);
				sumValues = sum(inputData.get(ref.get(i)), sumValues);
			}
			for (int i = 0; i < sumValues.size(); i++) {
				sumValuesbyn.add(sumValues.get(i) / ref.size());
			}
			clusterData.put(sumKey / ref.size(), sumValuesbyn);
		}
		return clusterData;
	}

	public List<Double> sum(List<Double> a, List<Double> b) {
		List<Double> sumValues = new ArrayList<>();
		for (int i = 0; i < a.size(); i++) {
			sumValues.add((a.get(i) + b.get(i)));
		}
		return sumValues;
	}

	// calculates the squared distance
	public double eucDistance(List<Double> p1, List<Double> p2) {
		Double dis = 0.0;
		for (int i = 0; i < p1.size(); i++) {
			dis += Math.pow(p1.get(i) - p2.get(i), 2);
		}
		return Math.sqrt(dis);
	}

	public static class InputMapper extends Mapper<Object, Text, Text, Text> {

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			String[] term = value.toString().split("\t");
			List<Double> list = new ArrayList<>();
			for (int i = 2; i < term.length; i++) {
				list.add(Double.parseDouble(term[i]));
			}
			inputData.put(Double.parseDouble(term[0]), list);
			groundTruth.put(Double.parseDouble(term[0]), Double.parseDouble(term[1]));

			context.write(new Text(""), new Text(""));

		}
	}

	// calculate initial centroid
	public static class InputReducer extends Reducer<Text, Text, Text, Text> {

		public void reduce(Text key, Iterable<Text> values, Context ctx) throws IOException, InterruptedException {
			final long startTime = System.currentTimeMillis();

			KMeans km = new KMeans();
			km.clusterPoints();
			while (!isMatch) {
				prevClusterList = new HashMap<Double, List<Double>>(clusterData);
				clusterData.keySet().removeAll(prevClusterList.keySet());
				prevClusterList.putAll(clusterData);
				// System.out.println("prevCluster: " + prevClusterList);

				km.updateClusterRef();
				// System.out.println("cluster ref map: " + clustRefMap);
				km.updateClusterPoints();

				for (Double current : clusterData.keySet()) {
					if (prevClusterList.containsKey(current))
						isMatch = true;
					else {
						isMatch = false;
						break;
					}
				}

			}
			System.out.println("cluster ref map: " + clustRefMap);
			 System.out.println("clusterData: " + clusterData);
//			System.out.println("inputData" + inputData);
			for (Double keys : groundTruth.keySet()) {
				StringBuilder out = new StringBuilder();
				for (Double clusRefKeys : clustRefMap.keySet()) {
					if (clustRefMap.get(clusRefKeys).contains(keys)) {
						out.append((int)Math.round(clusRefKeys)).append("\t").append((int)Math.round(groundTruth.get(keys)));
						System.out.println(keys + "\t" + out);
						ctx.write(new Text(String.valueOf((int)Math.round(keys))), new Text(out.toString()));
					}
				}
			}
			final long endTime = System.currentTimeMillis();
			System.out.println("Total execution time: " + (endTime - startTime));
		}
	}

	/**
	 * @param args
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		Configuration conf = new Configuration();
		Job job = Job.getInstance(conf, "kmeans");

		job.setJarByClass(KMeans.class);
		job.setMapperClass(InputMapper.class);

		job.setReducerClass(InputReducer.class);

		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));

		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
}