"""
Data preprocessing.
"""

import numpy as np
import json

class StandardScaler:
	"""
	An adapted implementation of the standard scaler to preprocess input and output data according to the type of fields.
	"""

	def __init__( self ):
		"""
		Constructor.
		"""
		self._meanPhi = 0.		# Mean of level-set values.
		self._stdPhi = 1.		# Standard deviation of level-set values.

		self._meanVel = 0.		# Mean of velocity components.
		self._stdVel = 1.		# Standard deviation of level-set values.

		self._meanDist = 0.		# Mean of distance between arrival and departure point.
		self._stdDist = 1.		# Standard deviation of distance.

		self._meanCoord = 0.	# Mean of scaled coordinates of departure point w.r.t. quad's lower corner.
		self._stdCoord = 1.		# Standard deviation of scaled coordinates.

		self._meanHK = 0.		# Mean of dimensionless curvature at the interface.
		self._stdHK = 1.		# Standard deviation of dimensionless curvature at the interface.

		self._meanPhi_xx = 0.	# Mean of h^2 scaled second spatial derivatives of level-set function.
		self._stdPhi_xx = 1.	# Standard deviation of second spatial derivatives of level-set function.


	def fit( self, inputs: np.ndarray ):
		"""
		Fit standard scaler object to properties of input data set.
		:param [in] inputs: Data set containing inputs (phi values, velocities, distances, coords) for inference model.
		"""
		assert len( inputs ) >= 1				# Expecting data.

		# Notice: No degrees of freedom when computing the standard deviation.

		# Collecting stats for phi values.
		self._meanPhi = np.mean( inputs[:, [0,6,7,8,9,21]] )					# Notice we are not computing stats along columns: it's for the whole
		self._stdPhi = np.std( inputs[:, [0,6,7,8,9,21]], ddof=1 )				# set.

		# Collecting stats for velocity components.
		self._meanVel = np.mean( inputs[:, [1,2,10,11,12,13,14,15,16,17]] )		# Not involving targets at all.
		self._stdVel = np.std( inputs[:, [1,2,10,11,12,13,14,15,16,17]], ddof=1 )

		# Collecting stats for distance variable.
		self._meanDist = np.mean( inputs[:, 3] )								# Not involving targets at all.
		self._stdDist = np.std( inputs[:, 3], ddof=1 )

		# Collecting stats for scaled coordinates of departure point w.r.t. owning quad's lower corner.
		self._meanCoord = np.mean( inputs[:, [4,5]] )							# Not involving targets at all.
		self._stdCoord = np.std( inputs[:, [4,5]], ddof=1 )

		# Collecting stats for scaled by h^2 second spatial derivatives of level-set function.
		self._meanPhi_xx = np.mean( inputs[:, [18,19]] )
		self._stdPhi_xx = np.std( inputs[:, [18,19]] )

		# Collecting stats for dimensionless curvature at the interface.
		self._meanHK = np.mean( inputs[:, 20] )
		self._stdHK = np.std( inputs[:, 20], ddof=1 )


	def transform( self, inputs: np.ndarray ) -> np.ndarray:
		"""
		Scale input values according to stored stats from training data set.
		:param [in] inputs: Data set containing inputs (phi values, velocities, distances, coords) for inference model.
		:return Transformed inputs.
		"""
		assert len( inputs ) >= 1					# Expecting data.

		inputs = np.copy( inputs ).astype( np.float64 )

		# Scaling phi values.
		inputs[:, [0,6,7,8,9,21]] = (inputs[:, [0,6,7,8,9,21]] - self._meanPhi) / self._stdPhi

		# Scaling velocity components.
		inputs[:, [1,2,10,11,12,13,14,15,16,17]] = (inputs[:, [1,2,10,11,12,13,14,15,16,17]] - self._meanVel) / self._stdVel

		# Scaling distance variable.
		inputs[:, 3] = (inputs[:, 3] - self._meanDist) / self._stdDist

		# Scaling departure point coords.
		inputs[:, [4,5]] = (inputs[:, [4,5]] - self._meanCoord) / self._stdCoord

		# Scaling second spatial derivative of level-set function.
		inputs[:, [18,19]] = (inputs[:, [18,19]] - self._meanPhi_xx) / self._stdPhi_xx

		# Scaling dimensionless curvature.
		inputs[:, 20] = (inputs[:, 20] - self._meanHK) / self._stdHK

		return inputs


	def fit_transform( self, inputs: np.ndarray ) -> np.ndarray:
		"""
		Fit transformer to training data and returned transformed inputs.
		:param [in] inputs: Data set containing inputs (phi values, velocities, distances, coords) for inference model.
		:return: Transformed inputs.
		"""
		self.fit( inputs )
		return self.transform( inputs )


	def toJSON( self, fileName="standardscaler.json" ):
		"""
		Convert and export properties of this transformer to JSON format for importing into C++.
		:param fileName JSON file name where to export data.
		"""
		obj = {
			"phi"      : {"mean":    self._meanPhi, "std":    self._stdPhi},
			"vel"      : {"mean":    self._meanVel, "std":    self._stdVel},
			"dist"     : {"mean":   self._meanDist, "std":   self._stdDist},
			"coord"    : {"mean":  self._meanCoord, "std":  self._stdCoord},
			"h2_phi_xx": {"mean": self._meanPhi_xx, "std": self._stdPhi_xx},
			"hk"       : {"mean":     self._meanHK, "std":     self._stdHK}
		}

		with open( fileName, "w" ) as f:
			json.dump( obj, f, sort_keys=True, indent=4 )