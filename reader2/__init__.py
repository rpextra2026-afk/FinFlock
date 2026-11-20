"""Reader v2 final: the hardened production reader built on the proven POC prompt (gate + worked examples).

It reuses the production worker (resume, cutoff, rolling checkpoints, one bundle) and adds the safety nets listed in
``reader2/session.py``. Nothing here changes the earlier ``reader`` job or its bundles.
"""
