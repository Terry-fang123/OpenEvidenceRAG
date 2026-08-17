import requests

def search_clinical_trials(keyword: str, max_results: int = 5):
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    params = {
        "query.term": keyword,
        "pageSize": max_results,
        "sort": "LastUpdatePostDate:desc",
    }

    headers = {
        "User-Agent": "Python ClinicalTrials Demo/1.0"
    }

    try:
        resp = requests.get(base_url, params=params, headers=headers)
        resp.raise_for_status()
        data = resp.json()

        study_list = data.get("studies", [])

        print(f"✅ 共检索到 {len(study_list)} 项相关临床试验：\n")

        for idx, item in enumerate(study_list, 1):
            protocol = item.get("protocolSection", {})
            
            identification = protocol.get("identificationModule", {})
            title = identification.get("briefTitle", "无标题")
            nct_id = identification.get("nctId", "无NCT ID")
            
            status = protocol.get("statusModule", {}).get("overallStatus", "状态未知")
            
            summary = protocol.get("descriptionModule", {}).get("briefSummary", "无简要描述")
            
            contacts_locations = protocol.get("contactsLocationsModule", {})
            locations = contacts_locations.get("locations", [])
            location_info = "无地点信息"
            if locations and isinstance(locations, list) and len(locations) > 0:
                if isinstance(locations[0], dict):
                    facility = locations[0].get("facility", {})
                    if isinstance(facility, dict):
                        location_info = facility.get("name", "无地点信息")
                    else:
                        location_info = str(facility)
                else:
                    location_info = str(locations[0])

            print(f"===== 第 {idx} 项研究 =====")
            print(f"NCT ID: {nct_id}")
            print(f"标题：{title}")
            print(f"状态：{status}")
            print(f"核心地点：{location_info}")
            print(f"简介：{summary[:200]}..." if len(summary) > 200 else f"简介：{summary}")
            print("-" * 80 + "\n")

        return data

    except requests.exceptions.RequestException as e:
        print(f"❌ 请求失败：{e}")
        return None

if __name__ == "__main__":
    search_clinical_trials(keyword="type 2 diabetes", max_results=3)