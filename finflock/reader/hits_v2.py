"""Point-in-time counterpart resolution; ambiguous and unknown names stay unresolved."""
from __future__ import annotations
import re
import pandas as pd
from finflock.data.sp500 import members_on


def normalise(text):
    return ' '.join(re.findall(r'\w+', str(text).casefold()))


class HitResolver:
    def __init__(self, aliases, snapshots, countries, commodities, sectors, prefixes):
        self.aliases, self.snapshots, self.prefixes = aliases.copy(), snapshots, prefixes
        self.aliases['_key'] = self.aliases['alias'].map(normalise)
        self.countries = self._table(countries, 'alias', 'iso')
        self.commodities = self._table(commodities, 'keyword', 'tag')
        self.sectors = {normalise(s): s for s in sectors['sector'].dropna().unique()}
        self._members = {}

    @staticmethod
    def _table(frame, name, node):
        result = {}
        for _, row in frame.iterrows():
            for alias in (row[name], row[node]):
                result.setdefault(normalise(alias), set()).add(str(row[node]))
        return result

    def resolve(self, hit, available_date, mask_mapping):
        result = {**hit, 'resolved': False, 'node_id': None, 'resolution_kind': 'unresolved'}
        role, name = hit['role'], hit['name']
        key = normalise(name)
        if role in ('country', 'commodity'):
            table = self.countries if role == 'country' else self.commodities
            nodes = table.get(key, set())
            if len(nodes) == 1:
                result.update(resolved=True, node_id=self.prefixes[role] + next(iter(nodes)), resolution_kind=role + '_gazetteer')
            elif len(nodes) > 1:
                result['resolution_kind'] = 'ambiguous'
        elif role == 'industry_group':
            words = set(key.split())
            matches = [v for k, v in self.sectors.items() if set(k.split()) <= words or words and words <= set(k.split())]
            if len(matches) == 1:
                # sic_sector defines sector categories, not asset graph nodes. Keep the distinction explicit.
                result.update(resolved=True, node_id='sector:' + matches[0], resolution_kind='sector_category')
            elif matches:
                result['resolution_kind'] = 'ambiguous'
        elif role in ('rival', 'supplier', 'customer'):
            date = pd.Timestamp(available_date).tz_localize(None).normalize()
            if date not in self._members:
                self._members[date] = members_on(self.snapshots, date).security_ids
            members = self._members[date]
            masked = {v: k for k, v in mask_mapping.items()}
            if name in masked:
                nodes = {masked[name]}
                kind = 'masked_pin'
            else:
                if re.fullmatch(r'COMPANY_[A-Z]+', name):
                    return result
                part = self.aliases[self.aliases['_key'] == key]
                start = pd.to_datetime(part['valid_from'], utc=True).dt.tz_localize(None)
                end = pd.to_datetime(part['valid_to'], utc=True).dt.tz_localize(None)
                active = part[(start.isna() | (start <= date)) & (end.isna() | (end >= date))]
                nodes, kind = set(active['node_id'].astype(str)), 'dated_alias'
            nodes &= set(members)
            if len(nodes) == 1:
                result.update(resolved=True, node_id=next(iter(nodes)), resolution_kind=kind)
            elif len(nodes) > 1:
                result['resolution_kind'] = 'ambiguous'
        return result

    def resolve_all(self, hits, available_date, mask_mapping):
        return [self.resolve(h, available_date, mask_mapping) for h in hits]
