import React from 'react';
import { motion } from 'framer-motion';
import { Layers, RotateCcw } from 'lucide-react';

const stepDescriptions = {
  original: 'Input image as captured by camera',
  normals: 'Estimated pseudo-3D surface normal map',
  perspective: '3D rotation simulating different viewing angles',
  lighting: 'Surface-aware Lambertian lighting model',
  elastic: 'Elastic deformation simulating skin stretch',
  color: 'HSV-space color / white-balance augmentation',
  augmented: 'Final augmented result for model training',
};

export default function AugmentationViewer({ augData, onRerun }) {
  if (!augData) return null;

  const { steps, step_order, step_labels } = augData;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-2xl space-y-4"
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers size={18} className="text-medical-400" />
          <h3 className="font-semibold text-white">3D Augmentation Pipeline</h3>
        </div>
        {onRerun && (
          <button
            onClick={onRerun}
            className="flex items-center gap-1.5 text-sm text-medical-400 hover:text-medical-300 transition-colors"
          >
            <RotateCcw size={14} />
            Re-augment
          </button>
        )}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {step_order.map((key, i) => (
          <motion.div
            key={key}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.07 }}
            className="glass-card overflow-hidden group"
          >
            <div className="aspect-square overflow-hidden">
              <img
                src={steps[key]}
                alt={step_labels[key]}
                className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
              />
            </div>
            <div className="p-2">
              <p className="text-xs font-semibold text-medical-300 truncate">
                {i + 1}. {step_labels[key]}
              </p>
              <p className="text-[10px] text-white/40 mt-0.5 line-clamp-2">
                {stepDescriptions[key]}
              </p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
