package main

import (
	"log"
	"net"
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	metricsapi "github.com/shajela/dto/api/metrics"
	"github.com/shajela/dto/pkg/metrics"
	"google.golang.org/grpc"
)

func main() {
	registry := prometheus.NewRegistry()
	store := metrics.NewMetricsStore(registry)

	// Start Prometheus HTTP endpoint
	go func() {
		http.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{}))
		log.Println("Prometheus scraping on :8080/metrics")
		log.Fatal(http.ListenAndServe(":8080", nil))
	}()

	// Start gRPC server
	lis, err := net.Listen("tcp", ":50051")
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	grpcServer := grpc.NewServer()
	metricsapi.RegisterMetricsServiceServer(grpcServer, &metrics.MetricsServer{Store: store})
	log.Println("gRPC server listening on :50051")
	log.Fatal(grpcServer.Serve(lis))
}
