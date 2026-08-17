from __future__ import annotations

import requests


CLINICAL_TRIALS_API_URL = (
    "https://clinicaltrials.gov/api/v2/studies"
)

CLINICAL_TRIAL_URL = "https://clinicaltrials.gov/study/{nct_id}"


def _get_module(
    study: dict,
    module_name: str,
) -> dict:
    protocol = study.get(
        "protocolSection",
        {},
    )

    module = protocol.get(
        module_name,
        {},
    )

    if isinstance(module, dict):
        return module

    return {}


def _join_values(
    values: list,
    max_items: int = 8,
) -> str:
    cleaned_values = [
        str(value).strip()
        for value in values[:max_items]
        if str(value).strip()
    ]

    return "; ".join(cleaned_values)


def _date_value(date_struct: dict) -> str:
    if not isinstance(date_struct, dict):
        return ""

    return str(
        date_struct.get("date", "")
    ).strip()


def _extract_conditions(study: dict) -> str:
    conditions_module = _get_module(
        study,
        "conditionsModule",
    )

    conditions = conditions_module.get(
        "conditions",
        [],
    )

    if isinstance(conditions, list):
        return _join_values(conditions)

    return ""


def _extract_interventions(study: dict) -> str:
    arms_module = _get_module(
        study,
        "armsInterventionsModule",
    )

    interventions = arms_module.get(
        "interventions",
        [],
    )

    if not isinstance(interventions, list):
        return ""

    intervention_texts: list[str] = []

    for intervention in interventions:
        if not isinstance(intervention, dict):
            continue

        name = str(
            intervention.get("name", "")
        ).strip()

        intervention_type = str(
            intervention.get("type", "")
        ).strip()

        if name and intervention_type:
            intervention_texts.append(
                f"{intervention_type}: {name}"
            )
        elif name:
            intervention_texts.append(name)

    return _join_values(intervention_texts)


def _extract_location(study: dict) -> str:
    contacts_module = _get_module(
        study,
        "contactsLocationsModule",
    )

    locations = contacts_module.get(
        "locations",
        [],
    )

    if not isinstance(locations, list):
        return ""

    location_texts: list[str] = []

    for location in locations[:5]:
        if not isinstance(location, dict):
            continue

        facility_value = location.get(
            "facility",
            "",
        )

        if isinstance(facility_value, dict):
            facility = facility_value.get(
                "name",
                "",
            )
        else:
            facility = facility_value

        city = location.get(
            "city",
            "",
        )

        country = location.get(
            "country",
            "",
        )

        parts = [
            str(part).strip()
            for part in [
                facility,
                city,
                country,
            ]
            if str(part).strip()
        ]

        if parts:
            location_texts.append(
                ", ".join(parts)
            )

    return _join_values(location_texts)


def _extract_phase(study: dict) -> str:
    design_module = _get_module(
        study,
        "designModule",
    )

    phases = design_module.get(
        "phases",
        [],
    )

    if isinstance(phases, list):
        return _join_values(phases)

    return ""


def _extract_enrollment(study: dict) -> str:
    design_module = _get_module(
        study,
        "designModule",
    )

    enrollment_info = design_module.get(
        "enrollmentInfo",
        {},
    )

    if not isinstance(enrollment_info, dict):
        return ""

    count = enrollment_info.get(
        "count",
        "",
    )

    enrollment_type = enrollment_info.get(
        "type",
        "",
    )

    if count and enrollment_type:
        return f"{count} ({enrollment_type})"

    if count:
        return str(count)

    return ""


def _extract_sponsor(study: dict) -> str:
    sponsors_module = _get_module(
        study,
        "sponsorCollaboratorsModule",
    )

    lead_sponsor = sponsors_module.get(
        "leadSponsor",
        {},
    )

    if not isinstance(lead_sponsor, dict):
        return ""

    return str(
        lead_sponsor.get("name", "")
    ).strip()


def parse_clinical_trial(study: dict) -> dict | None:
    """
    将 ClinicalTrials.gov v2 study 解析为可入库的结构化证据。
    """

    identification = _get_module(
        study,
        "identificationModule",
    )

    status_module = _get_module(
        study,
        "statusModule",
    )

    description_module = _get_module(
        study,
        "descriptionModule",
    )

    design_module = _get_module(
        study,
        "designModule",
    )

    nct_id = str(
        identification.get("nctId", "")
    ).strip()

    if not nct_id:
        return None

    title = str(
        identification.get("briefTitle", "")
        or identification.get("officialTitle", "")
    ).strip()

    status = str(
        status_module.get("overallStatus", "")
    ).strip()

    start_date = _date_value(
        status_module.get(
            "startDateStruct",
            {},
        )
    )

    completion_date = _date_value(
        status_module.get(
            "completionDateStruct",
            {},
        )
    )

    summary = str(
        description_module.get("briefSummary", "")
    ).strip()

    study_type = str(
        design_module.get("studyType", "")
    ).strip()

    conditions = _extract_conditions(study)
    interventions = _extract_interventions(study)
    phase = _extract_phase(study)
    enrollment = _extract_enrollment(study)
    sponsor = _extract_sponsor(study)
    location = _extract_location(study)
    url = CLINICAL_TRIAL_URL.format(nct_id=nct_id)

    evidence_text_parts = [
        f"NCT ID: {nct_id}",
        f"Title: {title}" if title else "",
        f"Status: {status}" if status else "",
        f"Study type: {study_type}" if study_type else "",
        f"Phase: {phase}" if phase else "",
        f"Conditions: {conditions}" if conditions else "",
        f"Interventions: {interventions}" if interventions else "",
        f"Enrollment: {enrollment}" if enrollment else "",
        f"Start date: {start_date}" if start_date else "",
        f"Completion date: {completion_date}" if completion_date else "",
        f"Sponsor: {sponsor}" if sponsor else "",
        f"Locations: {location}" if location else "",
        f"URL: {url}",
        f"Brief summary: {summary}" if summary else "",
    ]

    evidence_text = "\n".join(
        part
        for part in evidence_text_parts
        if part
    )

    return {
        "source_type": "clinical_trial",
        "nct_id": nct_id,
        "title": title,
        "status": status,
        "study_type": study_type,
        "phase": phase,
        "conditions": conditions,
        "interventions": interventions,
        "enrollment": enrollment,
        "start_date": start_date,
        "completion_date": completion_date,
        "sponsor": sponsor,
        "location": location,
        "url": url,
        "summary": summary,
        "text": evidence_text,
    }


def parse_clinical_trials(data: dict) -> list[dict]:
    """
    解析 ClinicalTrials.gov v2 API 返回值。
    """

    studies = data.get(
        "studies",
        [],
    )

    if not isinstance(studies, list):
        return []

    parsed_studies: list[dict] = []

    for study in studies:
        if not isinstance(study, dict):
            continue

        parsed_study = parse_clinical_trial(study)

        if parsed_study:
            parsed_studies.append(parsed_study)

    return parsed_studies


def fetch_clinical_trials(
    keyword: str,
    max_results: int = 5,
) -> list[dict]:
    """
    检索并解析 ClinicalTrials.gov 临床试验。
    """

    params = {
        "query.term": keyword,
        "pageSize": max_results,
        "sort": "LastUpdatePostDate:desc",
    }

    headers = {
        "User-Agent": "OpenEvidence Classroom MVP/1.0",
    }

    response = requests.get(
        CLINICAL_TRIALS_API_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    return parse_clinical_trials(
        response.json()
    )


def search_trials(keyword: str) -> list[str]:
    """
    兼容旧函数：只返回标题列表。
    """

    trials = fetch_clinical_trials(
        keyword=keyword,
        max_results=5,
    )

    return [
        trial["title"]
        for trial in trials
        if trial.get("title")
    ]


if __name__ == "__main__":
    for trial in fetch_clinical_trials(
        keyword="hypertension drug",
        max_results=3,
    ):
        print(trial)
