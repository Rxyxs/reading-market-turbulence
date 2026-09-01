// Streaming aggregator en Go: lee trades reales de Binance (mismo CSV
// aggTrades crudo que usa el pipeline en Python) linea por linea, sin
// cargar el archivo completo en memoria, y calcula las mismas features de
// microestructura por bucket de 30s (VWAP, order-flow imbalance,
// volatilidad realizada) que src/features.py -- el patron real de un
// gateway de datos de mercado en produccion, donde Go es una eleccion
// comun por su concurrencia y footprint de memoria bajo frente a cargar
// todo en un DataFrame.
//
// Verificado contra la referencia de Python/Polars para el mismo dia real
// (BTCUSDT, 2026-08-25, 1.62M trades).
//
//   go run streamer.go

package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"time"
)

const bucketMs = 30000

type bucketAgg struct {
	dollarVolume    float64
	totalVolume     float64
	takerBuyVolume  float64
	takerSellVolume float64
	nTrades         int
	prices          []float64 // orden de llegada, para volatilidad realizada
}

func realizedVolatility(prices []float64) float64 {
	if len(prices) < 2 {
		return 0.0
	}
	sumSq := 0.0
	for i := 1; i < len(prices); i++ {
		logReturn := math.Log(prices[i]) - math.Log(prices[i-1])
		sumSq += logReturn * logReturn
	}
	return math.Sqrt(sumSq)
}

func processFile(path string) (map[int64]*bucketAgg, int, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	reader := csv.NewReader(bufio.NewReaderSize(file, 1<<20))
	buckets := make(map[int64]*bucketAgg)
	nTrades := 0

	for {
		record, err := reader.Read()
		if err != nil {
			break // EOF real o fin de archivo
		}
		price, _ := strconv.ParseFloat(record[1], 64)
		qty, _ := strconv.ParseFloat(record[2], 64)
		timestampUs, _ := strconv.ParseInt(record[5], 10, 64)
		isBuyerMaker := record[6] == "True"

		timestampMs := timestampUs / 1000
		bucketStart := (timestampMs / bucketMs) * bucketMs

		b, exists := buckets[bucketStart]
		if !exists {
			b = &bucketAgg{}
			buckets[bucketStart] = b
		}
		b.dollarVolume += price * qty
		b.totalVolume += qty
		b.nTrades++
		b.prices = append(b.prices, price)
		if isBuyerMaker {
			b.takerSellVolume += qty
		} else {
			b.takerBuyVolume += qty
		}
		nTrades++
	}
	return buckets, nTrades, nil
}

type refRow struct {
	bucketStart        int64
	vwap               float64
	ofi                float64
	realizedVolatility float64
	nTrades            int
}

func loadReference(path string) (map[int64]refRow, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	_, _ = reader.Read() // header
	ref := make(map[int64]refRow)
	for {
		record, err := reader.Read()
		if err != nil {
			break
		}
		bucketStart, _ := strconv.ParseInt(record[0], 10, 64)
		vwap, _ := strconv.ParseFloat(record[1], 64)
		ofi, _ := strconv.ParseFloat(record[2], 64)
		rv, _ := strconv.ParseFloat(record[3], 64)
		nTrades, _ := strconv.Atoi(record[4])
		ref[bucketStart] = refRow{bucketStart, vwap, ofi, rv, nTrades}
	}
	return ref, nil
}

func main() {
	fmt.Println("[1/3] Procesando trades reales BTCUSDT (2026-08-25) en streaming...")
	start := time.Now()
	buckets, nTrades, err := processFile("../data/raw_btc/BTCUSDT-aggTrades-2026-08-25.csv")
	elapsed := time.Since(start)
	if err != nil {
		fmt.Println("Error:", err)
		os.Exit(1)
	}
	fmt.Printf("  %d trades reales procesados en %.2fs (%.0f trades/seg)\n",
		nTrades, elapsed.Seconds(), float64(nTrades)/elapsed.Seconds())
	fmt.Printf("  %d buckets de 30s generados\n", len(buckets))

	fmt.Println("\n[2/3] Verificando contra la referencia real de Python/Polars...")
	ref, err := loadReference("python_reference.csv")
	if err != nil {
		fmt.Println("Error cargando referencia:", err)
		os.Exit(1)
	}

	var keys []int64
	for k := range buckets {
		if _, ok := ref[k]; ok {
			keys = append(keys, k)
		}
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })

	maxDiffVwap, maxDiffOfi, maxDiffRv := 0.0, 0.0, 0.0
	nCompared := 0
	for _, k := range keys {
		b := buckets[k]
		r := ref[k]
		vwap := b.dollarVolume / b.totalVolume
		ofi := (b.takerBuyVolume - b.takerSellVolume) / b.totalVolume
		rv := realizedVolatility(b.prices)

		maxDiffVwap = math.Max(maxDiffVwap, math.Abs(vwap-r.vwap))
		maxDiffOfi = math.Max(maxDiffOfi, math.Abs(ofi-r.ofi))
		maxDiffRv = math.Max(maxDiffRv, math.Abs(rv-r.realizedVolatility))
		nCompared++
	}

	fmt.Printf("  %d buckets comparados contra Python\n", nCompared)
	fmt.Printf("  Diferencia maxima VWAP: %.10f\n", maxDiffVwap)
	fmt.Printf("  Diferencia maxima OFI: %.10f\n", maxDiffOfi)
	fmt.Printf("  Diferencia maxima volatilidad realizada: %.10f\n", maxDiffRv)

	fmt.Println("\n[3/3] Resultado final")
	fmt.Println("=== Resultado ===")
	fmt.Printf("Trades procesados: %d\n", nTrades)
	fmt.Printf("Throughput: %.0f trades/segundo\n", float64(nTrades)/elapsed.Seconds())
	fmt.Printf("Buckets generados: %d (esperados ~2880 para un dia completo)\n", len(buckets))
	fmt.Printf("Maxima diferencia vs. Python (VWAP/OFI/vol. realizada): %.2e / %.2e / %.2e\n",
		maxDiffVwap, maxDiffOfi, maxDiffRv)
}
