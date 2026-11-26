import React from 'react';
import { motion } from 'framer-motion';
import { Clock, Info, ChevronDown, ChevronUp } from 'lucide-react';

function ProbabilityBar({ label, probability, rank }) {
  const pct = (probability * 100).toFixed(1);

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
            className={`prob-bar h-full rounded-full bg-blue-500`}
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

  if (result.model_trained !== true) {
    return (
      <div className="glass-card p-5 w-full max-w-md" role="status">
        <h3 className="text-lg font-semibold text-white">No trained model is loaded.</h3>
        <p className="mt-2 text-sm text-white/60">
          Classification is unavailable. You can still explore the 3D surface and image augmentation.
        </p>
      </div>
    );
  }

  const topPredictions = expanded ? result.all_predictions : result.all_predictions.slice(0, 3);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-md space-y-4"
    >
      {/* Primary result card */}
      <div className="glass-card-solid p-5">
        <p className="text-xs text-white/40 mb-1">Experimental top class</p>
        <h3 className="text-lg font-bold text-white">{result.prediction}</h3>

        <div className="mt-4 pt-4 border-t border-white/10 grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-white/40 mb-0.5">Top Softmax Score</p>
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
          Class Scores (uncalibrated)
        </h4>
        <div className="space-y-3">
          {topPredictions.map((pred, i) => (
            <ProbabilityBar
              key={pred.class}
              label={pred.class}
              probability={pred.probability}
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
