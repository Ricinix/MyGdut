package com.example.mygdut.domain

import android.content.Context
import android.util.Log
import com.example.mygdut.R

class BuildingTransformer(context: Context) {
    private val campusNames = context.resources.getStringArray(R.array.campus_name)
    private val campusCodes = context.resources.getStringArray(R.array.campus_code)
    private val oneBuildingNames = context.resources.getStringArray(R.array.building_name_1)
    private val twoBuildingNames = context.resources.getStringArray(R.array.building_name_2)
    private val threeBuildingNames = context.resources.getStringArray(R.array.building_name_3)
    private val fourBuildingNames = context.resources.getStringArray(R.array.building_name_4)
    private val oneBuildingCodes = context.resources.getStringArray(R.array.building_code_1)
    private val twoBuildingCodes = context.resources.getStringArray(R.array.building_code_2)
    private val threeBuildingCodes = context.resources.getStringArray(R.array.building_code_3)
    private val fourBuildingCodes = context.resources.getStringArray(R.array.building_code_4)

    /**
     * 第一个是校区代码，第二个是教学楼代码
     */
    fun name2code(campusName: String, buildingName: String) : Pair<String, String>{
        Log.d(TAG, "campusName: $campusName, buildingName: $buildingName")
        val index = campusNames.indexOf(campusName)
        val campusCode = campusCodes[index]
        return when(index+1){
            1->{
                val bIndex = oneBuildingNames.indexOf(buildingName)
                campusCode to oneBuildingCodes[bIndex]
            }
            2->{
                val bIndex = twoBuildingNames.indexOf(buildingName)
                campusCode to twoBuildingCodes[bIndex]
            }
            3->{
                val bIndex = threeBuildingNames.indexOf(buildingName)
                campusCode to threeBuildingCodes[bIndex]
            }
            4->{
                val bIndex = fourBuildingNames.indexOf(buildingName)
                campusCode to fourBuildingCodes[bIndex]
            }
            else -> campusCode to ""
        }
    }

    companion object{
        private const val TAG = "BuildingResource"
    }

}