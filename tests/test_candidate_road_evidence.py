import unittest

from shapely.geometry import Point
from experiments.chronological.match_candidate_road_evidence import (
    audit_esri, direction_of_ramp, interpolate_bracket, interval_coverage,
    overlapping_events, project_xym, select_candidates,
)


class RoadEvidenceTests(unittest.TestCase):
    def test_xym_measure_is_not_length_or_z(self):
        g={'paths': [[[0,0,10], [100,0,10.01], [100,200,12.]] ]}
        fit=project_xym(Point(103,100),g)
        self.assertAlmostEqual(fit['distance_m'],3.)
        self.assertAlmostEqual(fit['chainage_m'],200.)
        self.assertAlmostEqual(fit['measure'],11.005)
        with self.assertRaisesRegex(ValueError,'Invalid XYM'):
            project_xym(Point(0,0),{'paths':[[[0,0,None],[1,1,1]]]})

    def test_direction_is_highway_not_cross_street(self):
        for text,direction in [('WBONFR NB WILLOW PASSRD','WB'),
                ('004/WB OFF TO BAILEY RD','WB'),('SEG EB ON FR NB PLEASANT HILL','EB'),
                ('NB OFF TO CLAYTON/MARKET','NB'),('UNKNOWN LOCATION',None)]:
            self.assertEqual(direction_of_ramp(text),direction)

    def test_prefix_reset_cannot_be_interpolated_or_extrapolated(self):
        def anchor(pm,chain,prefix='R'):
            return dict(pm=pm,chainage_m=chain,prefix=prefix,suffix='',
                        pm_route_id=prefix,path_index=0,objectid=int(chain))
        good=[anchor(1,0),anchor(1.1,160)]
        self.assertAlmostEqual(interpolate_bracket(good,80,'chainage_m','pm')['value'],1.05)
        for anchors,target in [(good,200),([anchor(1,0),anchor(1.1,160,'L')],80),
                               ([anchor(1,0),anchor(1.5,800)],80)]:
            with self.assertRaises(ValueError):
                interpolate_bracket(anchors,target,'chainage_m','pm')
        ambiguous=good+[anchor(2,0,'L'),anchor(2.1,160,'L')]
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            interpolate_bracket(ambiguous,80,'chainage_m','pm')

    def test_interval_join_has_positive_overlap_and_exposes_gaps(self):
        rows=[dict(routeid='A',dataitem='THROUGH_LANES',beginpoint=a,endpoint=b)
              for a,b in [(0,1),(1,2),(2.5,3),(3,4)]]
        matched=overlapping_events(rows,'A',1,3,'THROUGH_LANES')
        self.assertEqual(len(matched),2)
        self.assertAlmostEqual(interval_coverage(matched,1,3),.75)
        self.assertEqual(overlapping_events(rows,'B',1,3,'THROUGH_LANES'),[])

    def test_selection_uses_distinct_supported_prefiltered_roads(self):
        def row(road,support,passed='True'):
            return dict(road=road,both_report_supported_train_windows=str(support),
                metadata_prefilter_pass=passed,both_valid_unique_train_x_fraction='1',
                postmile_difference='.2',low_postmile_station_id='1',high_postmile_station_id='2')
        rows=[row('SR4-W',99),row('SR4-E',98),row('SR24-W',30),row('SR242-N',20),
              row('I80-E',0),row('I680-N',100,'False')]
        self.assertEqual([x['road'] for x in select_candidates(rows)],['SR4-W','SR24-W','SR242-N'])

    def test_unknown_crs_truncation_and_zm_encoding_fail(self):
        base=dict(spatialReference={'wkid':3310},hasM=True,hasZ=False,
                  features=[dict(geometry={'paths':[[[0,0,1],[1,1,2]]]})])
        self.assertEqual(audit_esri(base,'test')['m_vertices'],2)
        for obj in [dict(base,spatialReference={'wkid':4326}),
                    dict(base,exceededTransferLimit=True),dict(base,hasZ=True)]:
            with self.assertRaises(ValueError):
                audit_esri(obj,'test')


if __name__=='__main__':
    unittest.main()
