package main

import (
	"context"
	"flag"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/samber/lo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/klog/v2"
)

var (
	modeString string
	nodeName   string
	kubeconfig string
	benchmarks string
	threshold  int
)

func main() {
	flag.StringVar(&modeString, "mode", WithDefaultString("MODE", "precheck"), "mode, available values: bench, precheck")
	flag.StringVar(&nodeName, "node-name", WithDefaultString("NODE_NAME", ""), "name of node")
	flag.StringVar(&kubeconfig, "kubeconfig", WithDefaultString("KUBECONFIG", ""), "path of kubeconfig")
	flag.StringVar(&benchmarks, "benchmarks", WithDefaultString("BENCHMARKS", ""), "benchmarks to run, split with comma")
	flag.IntVar(&threshold, "threshold", WithDefaultInt("THRESHOLD", 80), "percent(0~100) threshold for precheck below which the bench will not pass")
	flag.Parse()

	mode := Mode(modeString)
	if !mode.Valid() {
		klog.Fatalf("invalid mode: %s", mode)
	}

	client := BuildClientOrDie(kubeconfig)
	node, err := client.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		klog.Fatalf("failed to get node: %v", err)
	}
	nodeBenchResults := make(map[string]map[string]float64)
	_ = json.Unmarshal([]byte(node.Annotations[LabelNodeBanchResults]), &nodeBenchResults)

	for _, benchmark := range strings.Split(benchmarks, ",") {
		if benchmark == "" {
			continue
		}
		cmd := exec.Command("bash", "-c", fmt.Sprintf("./%s.sh", benchmark))
		resultStr, err := cmd.CombinedOutput()
		if err != nil {
			klog.Fatalf("failed to run benchmark %s: %v", benchmark, err)
		}

		/*
			benchmark的执行结果示例:
			test1: 1.0
			test2: 2.0
		*/
		benchResults := make(map[string]float64)
		for _, line := range strings.Split(string(resultStr), "\n") {
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			parts := strings.SplitN(line, ":", 2)
			if len(parts) != 2 {
				klog.Errorf("invalid result line: %s, skip", line)
			}

			key := parts[0]
			result, err := strconv.ParseFloat(parts[1], 64)
			if err != nil {
				klog.Fatalf("failed to parse benchmark result %s: %v", resultStr, err)
			}
			benchResults[key] = result
			klog.Infof("[benchmark] %s %s: %f", benchmark, key, result)

			if mode == ModePrecheck {
				expectBenchResults, ok := nodeBenchResults[benchmark]
				if !ok {
					continue
				}
				expect, ok := expectBenchResults[key]
				if !ok {
					continue
				}
				if expect*float64(threshold)/100.0 > result {
					klog.Fatalf("[benchmark] %s expect %f with threshold %d, but got %f", benchmark, expect, threshold, result)
				}
			}
		}

		if mode == ModeBench {
			nodeBenchResults[benchmark] = benchResults
		}
	}

	if mode == ModeBench {
		b := lo.Must(json.Marshal(nodeBenchResults))
		patchBody := fmt.Sprintf(`{"metadata":{"annotations":{"%s":%q}}}`, LabelNodeBanchResults, string(b))
		if _, err := client.CoreV1().Nodes().Patch(context.TODO(), nodeName, types.StrategicMergePatchType, []byte(patchBody), metav1.PatchOptions{}); err != nil {
			klog.Fatalf("failed to patch node %s: %v", nodeName, err)
		}
	}
}
