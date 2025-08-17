package metrics

import (
	"io"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/shajela/dto/api/metrics"
	grpc "google.golang.org/grpc"
)

// In-memory store for metric collection
type MetricsStore struct {
	mu sync.Mutex
	// In-memory metric store
	// Decouple for aggregation
	// and transformation logic
	counters map[string]float64
	// Exposes in-memory counter to prometheus
	prometheusCounters map[string]prometheus.Gauge
	registry           *prometheus.Registry
}

func NewMetricsStore(r *prometheus.Registry) *MetricsStore {
	return &MetricsStore{
		counters:           make(map[string]float64),
		prometheusCounters: make(map[string]prometheus.Gauge),
		registry:           r,
	}
}

func (s *MetricsStore) Update(m *metrics.Metric) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Update in-memory counter
	s.counters[m.Name] += m.Value

	// Update prometheus gauge
	g, ok := s.prometheusCounters[m.Name]
	if !ok {
		g = prometheus.NewGauge(prometheus.GaugeOpts{
			Name: m.Name,
			Help: "Metric " + m.Name,
		})
		s.registry.MustRegister(g)
		s.prometheusCounters[m.Name] = g
	}
	g.Set(s.counters[m.Name])
}

type MetricsServer struct {
	metrics.UnimplementedMetricsServiceServer
	Store *MetricsStore
}

func (s *MetricsServer) StreamMetrics(stream grpc.ClientStreamingServer[metrics.Metric, metrics.MetricAck]) error {
	for {
		m, err := stream.Recv()

		if err == io.EOF {
			return stream.SendAndClose(&metrics.MetricAck{Success: true})
		}

		if err != nil {
			return err
		}

		s.Store.Update(m)
	}
}
