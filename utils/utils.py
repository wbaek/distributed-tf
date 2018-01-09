import tensorflow as tf
from tensorflow.python.client import timeline

import logging
logger = logging.getLogger(__name__)


def run_session_with_profile(sess, fetches, profile_dir='./profile/temp/'):
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    results = sess.run(fetches, options=options, run_metadata=run_metadata)

    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
    chrome_trace = fetched_timeline.generate_chrome_trace_format()

    os.makedirs(profile_dir, exist_ok=True)
    with open('%s/timeline_%04d.json' % (profile_dir, results['global_step']), 'w') as f:
        f.write(chrome_trace)
    return results
