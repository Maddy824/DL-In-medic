import React from 'react';
import { motion } from 'framer-motion';
import {
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Clock,
  Activity,
  Info,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

const riskConfig = {
  critical: { icon: ShieldAlert, color: 'red', label: 'CRITICAL', bg: 'risk-critical' },
  high: { icon: AlertTriangle, color: 'orange', label: 'HIGH RISK', bg: 'risk-high' },
  moderate: { icon: Activity, color: 'yellow', label: 'MODERATE', bg: 'risk-moderate' },
  low: { icon: ShieldCheck, color: 'green', label: 'LOW RISK', bg: 'risk-low' },
};

function ProbabilityBar({ label, probability, riskLevel, rank }) {
  const pct = (probability * 100).toFixed(1);
  const risk = riskConfig[riskLevel] || riskConfig.low;
  const barColors = {
    critical: 'bg-red-500',
    high: 'bg-orange-500',
    moderate: 'bg-yellow-500',
    low: 'bg-green-500',
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: rank * 0.08 }}
      className="flex items-center gap-3"
    >
      <div className="flex-1 min-w-0">
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-medium text-white/80 truncate">{label}</span>
          <span className="text-sm font-mono text-white/60">{pct}%</span>
        </div>
        <div className="h-2 bg-white/5 rounded-full overflow-hidden">
          <div
            className={`prob-bar h-full rounded-full ${barColors[riskLevel] || 'bg-blue-500'}`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </motion.div>
  );
}

export default function Results({ result }) {
  const [expanded, setExpanded] = React.useState(false);

  if (!result) return null;

  const risk = riskConfig[result.risk_level] || riskConfig.low;
  const RiskIcon = risk.icon;
  const topPredictions = expanded ? result.all_predictions : result.all_predictions.slice(0, 3);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-md space-y-4"
    >
      {/* Primary result card */}
      <div className={`glass-card-solid p-5 border ${risk.bg}`}>
        <div className="flex items-start gap-4">
          <div className={`p-3 rounded-xl bg-${risk.color}-500/20`}>
            <RiskIcon size={28} className={`text-${risk.color}-400`} />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className={`text-xs font-bold tracking-wider px-2 py-0.5 rounded-full bg-${risk.color}-500/20 text-${risk.color}-300`}>
                {risk.label}
              </span>
            </div>
            <h3 className="text-lg font-bold text-white mb-1">{result.prediction}</h3>
            <p className="text-sm text-white/60 leading-relaxed">{result.description}</p>
          </div>
        </div>

        {/* Confidence metrics */}
        <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-white/40 mb-0.5">Top Probability</p>
            <p className="text-xl font-bold font-mono text-white">
              {(result.top_probability * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-white/40 mb-0.5">Inference Time</p>
            <p className="text-xl font-bold font-mono text-white flex items-center gap-1">
              <Clock size={16} className="text-white/40" />
              {result.inference_time_ms}ms
            </p>
          </div>
        </div>
      </div>

      {/* All predictions breakdown */}
      <div className="glass-card p-5">
        <h4 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">
          Classification Breakdown
        </h4>
        <div className="space-y-3">
          {topPredictions.map((pred, i) => (
            <ProbabilityBar
              key={pred.class}
              label={pred.class}
              probability={pred.probability}
              riskLevel={pred.risk_level}
              rank={i}
            />
          ))}
        </div>
        {result.all_predictions.length > 3 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-3 flex items-center gap-1 text-sm text-medical-400 hover:text-medical-300 transition-colors"
          >
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            {expanded ? 'Show less' : `Show all ${result.all_predictions.length} classes`}
          </button>
        )}
      </div>

      {/* Disclaimer */}
      <div className="flex items-start gap-2 p-3 rounded-xl bg-amber-500/5 border border-amber-500/20">
        <Info size={16} className="text-amber-400 mt-0.5 shrink-0" />
        <p className="text-xs text-amber-200/70 leading-relaxed">
          {result.disclaimer}
        </p>
      </div>
    </motion.div>
  );
}
